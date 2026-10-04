"""TbcclPipelineComm: exo's pipeline communication contract implemented over the TBCCL C ABI v1.

Nothing here interprets dtypes: every operation is byte movement. Reductions are not used; the one agreement exo needs (``any_true``) is an
all-gather of one byte per rank followed by a local OR.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from . import _cuda_caps
from ._loader import native
from .bridge import TRACE, Borrow, CopyStats, borrow
from .recv_pool import ReceivePool
from .config import MANAGED_AUTO, MANAGED_CUDA, MANAGED_HOST, FastPathConfig
from .errors import TbcclAbortedError, TbcclError, TbcclInvalidArgumentError, error_for

if TYPE_CHECKING:
    import mlx.core as mx

log = logging.getLogger("exo_tbccl")

ByteExchange = Callable[[str, bytes], Sequence[bytes]]
"""All-gather of opaque bytes supplied by the host application: ``exchange(purpose, payload)`` returns every rank's payload in rank order."""


_PROVEN_MANAGED_HOST_SIGNATURES = (
    # Linux, discrete GPU, driver-managed pageable access without host page tables: CPU access to managed pages is legal while the GPU runs and
    # faults the pages to the host (RTX 3070 Ti Laptop, sm_86, driver 610.43.02, Phase 54). Add a signature only after the stress gates pass on it.
    {"managed_memory": 1, "concurrent_managed_access": 1, "pageable_memory_access": 1, "pageable_memory_access_uses_host_page_tables": 0,
     "direct_managed_mem_access_from_host": 0, "integrated": 0},
)


def _AUTO_ALLOWED(caps: "_cuda_caps.DeviceCaps") -> bool:
    """True when the driver-reported attributes equal a signature the Phase 54 capability audit proved safe for host-direct managed access."""
    return any(all(caps.attrs.get(k) == v for k, v in sig.items()) for sig in _PROVEN_MANAGED_HOST_SIGNATURES)


class Transfer:
    """One submitted TBCCL operation plus everything that must stay alive until it is terminal."""

    __slots__ = ("work", "borrows", "op", "peer", "_terminal", "detached")

    def __init__(self, work: object, borrows: Sequence[Borrow], op: str, peer: int | None):
        self.detached = False  # an asynchronous send nobody waits on: its failure is held and raised at the next communication point
        self.work = work
        self.borrows = tuple(borrows)
        self.op = op
        self.peer = peer
        self._terminal = False

    def _finish(self) -> None:
        if not self._terminal:
            self._terminal = True
            for b in self.borrows:
                b.release()

    @property
    def terminal(self) -> bool:
        return self._terminal


class TbcclPipelineComm:
    def __init__(self, comm: object, rank: int, size: int, config: FastPathConfig | None = None):
        self.config = config or FastPathConfig.from_env()
        self._managed_verified: dict[int, bool] = {}
        poison = os.environ.get("EXO_TBCCL_RECV_POISON")
        self.pool = ReceivePool(poison=int(poison, 0) & 0xFF if poison else None)
        self._deferred_error: TbcclError | None = None
        self._async_sends: deque[Transfer] = deque()
        self._comm = comm
        self._rank = rank
        self._size = size
        self._lock = threading.Lock()
        self._pending: set[Transfer] = set()
        self._closed = False
        self.stats = CopyStats()
        self._cuda = native.register_cuda() == 0

    # ---- construction -----------------------------------------------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        rank: int,
        world_size: int,
        exchange: ByteExchange,
        *,
        bind_host: str | None = None,
        advertise_host: str | None = None,
        timeout_ms: int = 0,
        config: FastPathConfig | None = None,
    ) -> "TbcclPipelineComm":
        """UniqueId exchange -> BootstrapBegin -> endpoint blob all-gather -> BootstrapComplete. ``exchange`` is the host's opaque all-gather."""
        uids = exchange("uid", native.unique_id())
        if len(uids) != world_size or any(len(u) != 16 for u in uids):
            raise TbcclInvalidArgumentError(native.TBCCL_INVALID_ARGUMENT, "bootstrap", "unique-id exchange returned malformed data")
        bs = native.Bootstrap(rank, world_size, bytes(uids[0]), bind_host, advertise_host, timeout_ms)
        try:
            blobs = exchange("endpoint", bs.endpoint())
            if len(blobs) != world_size or any(len(b) != native.TBCCL_ENDPOINT_BLOB_SIZE for b in blobs):
                raise TbcclInvalidArgumentError(native.TBCCL_INVALID_ARGUMENT, "bootstrap", "endpoint exchange returned malformed data")
            comm = bs.complete(b"".join(bytes(b) for b in blobs))
        except TbcclError as e:
            raise e.with_context(rank=rank) from None
        finally:
            bs.close()
        return cls(comm, rank, world_size, config)

    # ---- helpers ----------------------------------------------------------------------------------------------------------------------

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size

    def _managed_as_host(self, send: bool, recv: bool, array: "mx.array") -> bool:
        """Policy for describing kDLCUDAManaged storage to TBCCL as host memory. ``cuda`` (the default) never does; ``host`` is a forced override
        for measurement; ``auto`` needs the array to be CUDA-managed, its device to match a capability signature the Phase 54 audit proved
        (docs/cuda_managed_memory.md), and the payload to be small enough that CPU access to GPU-written pages stays cheap. Anything else,
        including a driver that cannot be queried, keeps the CUDA path."""
        cfg = self.config
        if cfg.managed_mode == MANAGED_CUDA:
            return False
        if (send and not cfg.managed_send) or (recv and not cfg.managed_recv):
            return False
        if cfg.managed_mode == MANAGED_HOST:
            return True
        if array.nbytes > cfg.managed_max_bytes:
            return False
        device_type, ordinal = array.__dlpack_device__()
        if device_type != native.DL_CUDA_MANAGED:
            return False
        if ordinal not in self._managed_verified:
            caps = _cuda_caps.device_caps(ordinal)
            self._managed_verified[ordinal] = bool(caps and _AUTO_ALLOWED(caps))
        return self._managed_verified[ordinal]

    def _check_open(self) -> None:
        if self._closed:
            raise TbcclAbortedError(native.TBCCL_ABORTED, "comm", "communicator is closed", rank=self._rank)

    def _track(self, t: Transfer) -> Transfer:
        with self._lock:
            self._pending.add(t)
        return t

    def _reap(self) -> None:
        """Release every terminal transfer (non-blocking). A failed detached send is held and raised by ``_raise_deferred``."""
        with self._lock:
            pending = list(self._pending)
        for t in pending:
            if t.terminal:
                continue
            done, result = t.work.test()  # pyright: ignore[reportAttributeAccessIssue]
            if done:
                self._settle(t, result)
        self._raise_deferred()

    def _settle(self, t: Transfer, result: int) -> None:
        t._finish()
        with self._lock:
            self._pending.discard(t)
        if t.detached:
            try:
                self._async_sends.remove(t)
            except ValueError:
                pass
            self.stats.async_send_reaped += 1
            if result != native.TBCCL_SUCCESS and self._deferred_error is None:
                self._deferred_error = error_for(result, t.op, t.work.error_string()).with_context(rank=self._rank, peer=t.peer)  # pyright: ignore[reportAttributeAccessIssue]

    def _raise_deferred(self) -> None:
        err, self._deferred_error = self._deferred_error, None
        if err is not None:
            raise err

    def _drain_async_sends(self) -> None:
        """A synchronizing point (after a collective): wait for every still-outstanding detached send, then raise a held failure."""
        for t in list(self._async_sends):
            try:
                done, result = t.work.wait(None)  # pyright: ignore[reportAttributeAccessIssue]
            except TbcclError as e:
                self._settle(t, native.TBCCL_TRANSPORT_ERROR)
                self._deferred_error = self._deferred_error or e.with_context(rank=self._rank, peer=t.peer)
                continue
            if done:
                self._settle(t, result)
        self._raise_deferred()

    def _trace(self, op: str, peer: int | None, b: Borrow) -> None:
        self.stats.note_direct(b.label, b.nbytes)
        if TRACE:
            log.info("rank %d %s peer=%s bytes=%d path=%s", self._rank, op, peer, b.nbytes, b.label)

    def _submit(self, op: str, peer: int | None, borrows: Sequence[Borrow], call: Callable[[], object]) -> Transfer:
        self._check_open()
        try:
            work = call()
        except TbcclError as e:
            for b in borrows:
                b.release()
            raise e.with_context(rank=self._rank, peer=peer) from None
        return self._track(Transfer(work, borrows, op, peer))

    def wait(self, t: Transfer, timeout_ms: int | None = None) -> bool:
        """Wait for one transfer (GIL released). Returns True when terminal; raises the structured error of a failed operation."""
        try:
            done, result = t.work.wait(timeout_ms)  # pyright: ignore[reportAttributeAccessIssue]
        except TbcclError as e:
            raise e.with_context(rank=self._rank, peer=t.peer) from None
        if not done:
            return False
        t._finish()
        with self._lock:
            self._pending.discard(t)
        if result != native.TBCCL_SUCCESS:
            raise error_for(result, t.op, t.work.error_string()).with_context(rank=self._rank, peer=t.peer)  # pyright: ignore[reportAttributeAccessIssue]
        return True

    def wait_all(self, transfers: Sequence[Transfer]) -> None:
        """Wait for every transfer even after a failure (so no borrow outlives its Work), then raise the first error."""
        first: TbcclError | None = None
        for t in transfers:
            try:
                self.wait(t)
            except TbcclError as e:
                first = first or e
        if first is not None:
            raise first

    # ---- operations -------------------------------------------------------------------------------------------------------------------

    def send_async(self, array: "mx.array", dst: int) -> Transfer:
        self._reap()
        b = borrow(array, self.stats, managed_as_host=self._managed_as_host(True, False, array))
        self.pool.pin_aliased(b.ptr, b.nbytes)
        self._trace("send", dst, b)
        return self._submit("send", dst, [b], lambda: self._comm.send(b.ptr, b.nbytes, b.kind, b.device, dst))  # pyright: ignore[reportAttributeAccessIssue]

    def recv_into_async(self, dest: "mx.array", src: int) -> Transfer:
        self._reap()
        b = borrow(dest, self.stats, writable=True, managed_as_host=self._managed_as_host(False, True, dest))
        self._trace("recv", src, b)
        return self._submit("recv", src, [b], lambda: self._comm.recv(b.ptr, b.nbytes, b.kind, b.device, src))  # pyright: ignore[reportAttributeAccessIssue]

    def send(self, array: "mx.array", dst: int) -> "mx.array":
        if self.config.async_send:
            t = self.send_async(array, dst)
            t.detached = True
            self._async_sends.append(t)
            self.stats.async_send_submitted += 1
            return array
        self.wait(self.send_async(array, dst))
        return array

    def flush_sends(self, sends: Sequence[tuple["mx.array", int]]) -> None:
        """Submit every queued send first, then wait as a group (the nonblocking-submission contract)."""
        transfers: list[Transfer] = []
        try:
            for array, dst in sends:
                transfers.append(self.send_async(array, dst))
        except TbcclError:
            self.wait_all(transfers)
            raise
        self.wait_all(transfers)

    def recv_like(self, template: "mx.array", src: int) -> "mx.array":
        import mlx.core as mx

        if not self.config.recv_reuse:
            dest = mx.zeros(template.shape, dtype=template.dtype)
            mx.eval(dest)
            self.wait(self.recv_into_async(dest, src))
            return dest
        slot = self.pool.acquire(tuple(template.shape), template.dtype)
        t = self.recv_into_async(slot.array, src)
        ptr = t.borrows[0].ptr
        try:
            self.wait(t)
        except BaseException:
            self.pool.discard(slot)
            raise
        self.pool.mark_received(slot, ptr)
        return slot.array

    def step_complete(self) -> None:
        """exo's lifetime boundary: the stage output of the current forward step has been evaluated, so every consumer of the activations
        received for this step has completed and their destinations may be reused."""
        self.pool.release_leased()

    def all_gather(self, array: "mx.array") -> "mx.array":
        import mlx.core as mx

        self._reap()
        shape = tuple(array.shape)
        out_shape = (self._size,) if len(shape) == 0 else (self._size * shape[0], *shape[1:])
        dest = mx.zeros(out_shape, dtype=array.dtype)
        mx.eval(dest)
        as_host = self._managed_as_host(True, True, array)
        sb = borrow(array, self.stats, managed_as_host=as_host)
        try:
            rb = borrow(dest, self.stats, writable=True, managed_as_host=as_host)
        except TbcclError:
            sb.release()
            raise
        self._trace("all_gather", None, sb)
        t = self._submit("all_gather", None, [sb, rb], lambda: self._comm.all_gather(sb.ptr, sb.nbytes, rb.ptr, rb.nbytes, sb.kind, sb.device))  # pyright: ignore[reportAttributeAccessIssue]
        self.wait(t)
        self._drain_async_sends()
        return dest

    def barrier(self) -> None:
        self._reap()
        self.wait(self._submit("barrier", None, [], lambda: self._comm.barrier()))  # pyright: ignore[reportAttributeAccessIssue]
        self._drain_async_sends()

    def any_true(self, value: bool) -> bool:
        """All-gather of one byte per rank, then a local OR (exact, no reduction collective)."""
        self._reap()
        send = (ctypes.c_uint8 * 1)(1 if value else 0)
        recv = (ctypes.c_uint8 * self._size)()
        sb = Borrow(send, None, None, ctypes.addressof(send), 1, native.TBCCL_MEMORY_HOST, -1, "host")
        rb = Borrow(recv, None, None, ctypes.addressof(recv), self._size, native.TBCCL_MEMORY_HOST, -1, "host")
        t = self._submit("any_true", None, [sb, rb], lambda: self._comm.all_gather(sb.ptr, 1, rb.ptr, self._size, native.TBCCL_MEMORY_HOST, -1))  # pyright: ignore[reportAttributeAccessIssue]
        self.wait(t)
        self._drain_async_sends()
        return any(recv)

    # ---- lifecycle --------------------------------------------------------------------------------------------------------------------

    def abort(self, reason: str = "aborted") -> None:
        if not self._closed:
            self._comm.abort(reason)  # pyright: ignore[reportAttributeAccessIssue]

    def is_aborted(self) -> bool:
        return self._closed or bool(self._comm.is_aborted())  # pyright: ignore[reportAttributeAccessIssue]

    def close(self) -> None:
        """Release every borrow and destroy the communicator. Idempotent.

        With work still in flight the communicator is aborted first (so every Work becomes terminal and the drain is bounded). With nothing in
        flight no abort is sent: an abort is communicator-wide and would fail a peer that is still completing its side of the last collective.
        """
        if self._closed:
            return
        self._closed = True
        self._reap_for_close()
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        if pending:
            try:
                self._comm.abort("closing")  # pyright: ignore[reportAttributeAccessIssue]
            except TbcclError:
                pass
        for t in pending:
            try:
                t.work.wait()  # pyright: ignore[reportAttributeAccessIssue]
            except TbcclError:
                pass
            t._finish()
        self._async_sends.clear()
        self.pool.clear()
        self._comm.close()  # pyright: ignore[reportAttributeAccessIssue]

    def _reap_for_close(self) -> None:
        with self._lock:
            pending = list(self._pending)
        for t in pending:
            if t.terminal:
                continue
            try:
                done, _ = t.work.test()  # pyright: ignore[reportAttributeAccessIssue]
            except TbcclError:
                continue
            if done:
                t._finish()
                with self._lock:
                    self._pending.discard(t)

    def __enter__(self) -> "TbcclPipelineComm":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
