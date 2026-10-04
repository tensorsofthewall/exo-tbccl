"""Bounded pool of pipeline receive destinations (Phase 54).

recv_like used to allocate and evaluate a fresh MLX destination for every receive (~130-165 us on Metal, ~170-280 us on CUDA). A slot is reusable only
once exo says the stage's input activation is dead (``release_leased``, called from ``TbcclPipelineComm.step_complete`` after the stage output has been
evaluated); nothing here relies on Python reference counts. Slots are keyed by (dtype, shape), never by byte count.

States: FREE (cached, unused) -> RECEIVING (a TBCCL recv is writing it) -> LEASED (handed to MLX) -> FREE (at step_complete). A failed receive discards
the slot. Memory is bounded: at most ``depth`` FREE slots per key, ``max_cached_bytes`` over all tracked slots (FREE evicted least-recently-used
first), and at most ``max_leased`` LEASED slots; a slot beyond those is simply not tracked (never reused, freed by MLX when exo drops it).
"""

from __future__ import annotations

import ctypes
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

FREE, RECEIVING, LEASED = "FREE", "RECEIVING", "LEASED"


@dataclass(eq=False)
class Slot:
    key: tuple
    array: "mx.array"
    nbytes: int
    state: str = FREE
    tracked: bool = True
    ptr: int = 0


@dataclass
class PoolStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    untracked: int = 0
    peak_bytes: int = 0
    peak_slots: int = 0


@dataclass
class ReceivePool:
    depth: int = 2
    max_cached_bytes: int = 64 << 20
    max_leased: int = 4
    poison: int | None = None
    stats: PoolStats = field(default_factory=PoolStats)

    def __post_init__(self) -> None:
        self._free: OrderedDict[Slot, None] = OrderedDict()  # LRU order, oldest first
        self._leased: list[Slot] = []
        self._receiving: set[Slot] = set()

    # ---- accounting -------------------------------------------------------------------------------------------------------------------

    @property
    def cached_bytes(self) -> int:
        return sum(s.nbytes for s in self._free) + sum(s.nbytes for s in self._leased) + sum(s.nbytes for s in self._receiving)

    @property
    def slot_count(self) -> int:
        return len(self._free) + len(self._leased) + len(self._receiving)

    def _note_peak(self) -> None:
        self.stats.peak_bytes = max(self.stats.peak_bytes, self.cached_bytes)
        self.stats.peak_slots = max(self.stats.peak_slots, self.slot_count)

    # ---- lifecycle --------------------------------------------------------------------------------------------------------------------

    def acquire(self, shape: tuple[int, ...], dtype: "mx.Dtype") -> Slot:
        import mlx.core as mx

        key = (dtype, tuple(shape))
        for s in self._free:
            if s.key == key:
                del self._free[s]
                s.state = RECEIVING
                self._receiving.add(s)
                self.stats.hits += 1
                return s
        self.stats.misses += 1
        arr = mx.zeros(shape, dtype=dtype)
        mx.eval(arr)
        s = Slot(key, arr, arr.nbytes, RECEIVING)
        if len(self._receiving) + len(self._leased) >= self.max_leased or arr.nbytes > self.max_cached_bytes:
            s.tracked = False
            self.stats.untracked += 1
            return s
        self._receiving.add(s)
        self._evict()
        self._note_peak()
        return s

    def mark_received(self, slot: Slot, ptr: int) -> None:
        slot.ptr = ptr
        if slot.tracked:
            self._receiving.discard(slot)
            slot.state = LEASED
            self._leased.append(slot)

    def pin_aliased(self, ptr: int, nbytes: int) -> None:
        """A stage output that aliases a leased activation (an identity stage) must not be overwritten by a later receive: stop pooling that slot."""
        for s in list(self._leased):
            if s.ptr < ptr + nbytes and ptr < s.ptr + s.nbytes:
                self.discard(s)

    def discard(self, slot: Slot) -> None:
        self._receiving.discard(slot)
        if slot in self._leased:
            self._leased.remove(slot)
        slot.tracked = False

    def release_leased(self) -> None:
        """exo's boundary: every consumer of the leased activations has completed. LEASED -> FREE (poisoned first in the negative-control mode)."""
        leased, self._leased = self._leased, []
        for s in leased:
            if self.poison is not None:
                self._poison(s)
            s.state = FREE
            same = [f for f in self._free if f.key == s.key]
            if len(same) >= self.depth:
                self._free.pop(same[0])
                self.stats.evictions += 1
            self._free[s] = None
        self._evict()

    def clear(self) -> None:
        self._free.clear()
        self._leased.clear()
        self._receiving.clear()

    def _evict(self) -> None:
        while self._free and self.cached_bytes > self.max_cached_bytes:
            self._free.popitem(last=False)
            self.stats.evictions += 1

    def _poison(self, slot: Slot) -> None:
        import mlx.core as mx

        from ._loader import native

        uint = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[slot.array.dtype.size]
        view = slot.array.view(uint)
        mx.eval(view)
        exp = native.Export(view)
        try:
            ctypes.memset(exp.ptr, self.poison, exp.nbytes)
        finally:
            exp.release()
