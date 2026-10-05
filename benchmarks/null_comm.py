"""Phase 57 TEST-ONLY control: a PipelineComm whose cross-rank transport is the cheapest correct local exchange available to the harness
(POSIX shared memory plus a spun sequence word) so that what remains is the pipeline's own synchronization and evaluation pattern.

It keeps exo's required call semantics (send returns an array to depend on; recv_like returns a materialized array of the template's shape/dtype;
all_gather concatenates in rank order; step_complete exists) and exchanges real data, so the model's tokens stay identical. It is NOT a production
backend and is never imported by exo or exo_tbccl: only benchmarks/real_model_loopback.py (`--backend null`) loads it.

Per payload it does the minimum a host must: evaluate the array, copy its bytes into shared memory (numpy view of the same-width unsigned alias),
and on the receiving side build an mx.array from the bytes. Compare MlxRing / TbcclPipelineComm / null: if TbcclPipelineComm stays much slower
than null with essentially free transport, the bridge/synchronization is the cost; if not, it is not.
"""
import time
from multiprocessing import shared_memory

import numpy as np

SLOT = 8 << 20  # 8 MiB per slot (a 2048-token prefill chunk of hidden 1024 bf16 is 4 MiB)
HDR = 64


class _Slot:
    """One-directional mailbox: [0:8] sequence (writer bumps), [8:16] consumed sequence (reader bumps), [16:24] length."""

    def __init__(self, shm):
        self.shm = shm
        self.u64 = np.frombuffer(shm.buf, dtype=np.uint64, count=HDR // 8)
        self.data = np.frombuffer(shm.buf, dtype=np.uint8, count=SLOT, offset=HDR)
        self.seq = 0

    def put(self, raw: np.ndarray) -> None:
        while self.u64[0] != self.u64[1]:  # previous message not yet consumed
            time.sleep(0)
        self.data[: raw.nbytes] = raw
        self.u64[2] = raw.nbytes
        self.u64[0] = self.u64[0] + 1

    def take(self) -> np.ndarray:
        while self.u64[0] == self.u64[1]:
            time.sleep(0)
        out = self.data[: int(self.u64[2])].copy()
        self.u64[1] = self.u64[1] + 1
        return out


class NullPipelineComm:
    def __init__(self, rank, size, mailboxes, owned):
        self._rank, self._size = rank, size
        self._box = mailboxes  # (src, dst) -> _Slot, plus ("ag", rank) -> _Slot
        self._owned = owned

    @classmethod
    def create(cls, rank, world, exchange):
        names = {}
        owned = []
        keys = [(s, d) for s in range(world) for d in range(world) if s != d] + [("ag", r) for r in range(world)]
        mine = []
        if rank == 0:
            for k in keys:
                shm = shared_memory.SharedMemory(create=True, size=HDR + SLOT)
                shm.buf[:HDR] = bytes(HDR)
                owned.append(shm)
                names[k] = shm.name
                mine.append((k, shm.name))
        blob = repr(mine).encode() if rank == 0 else b""
        got = exchange("null", blob)
        if rank != 0:
            mine = eval(bytes(got[0]).decode())  # noqa: S307 - harness-local, our own bytes
            for k, name in mine:
                names[k] = name
                owned.append(None)
        boxes = {}
        for k, name in names.items():
            shm = owned[list(names).index(k)] if rank == 0 else shared_memory.SharedMemory(name=name)
            if rank != 0:
                owned[list(names).index(k)] = shm
            boxes[k] = _Slot(shm)
        return cls(rank, world, boxes, owned)

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    @staticmethod
    def _raw(array):
        import mlx.core as mx

        mx.eval(array)
        uint = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[array.dtype.size]
        v = array.view(uint)
        mx.eval(v)
        return np.ascontiguousarray(np.asarray(v)).view(np.uint8).reshape(-1)

    @staticmethod
    def _from_raw(raw, shape, dtype):
        import mlx.core as mx

        uint = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[dtype.size]
        mx_uint = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[dtype.size]
        return mx.array(raw.view(uint).reshape(shape)).view(dtype) if dtype != mx_uint else mx.array(raw.view(uint).reshape(shape))

    def send(self, array, dst):
        self._box[(self._rank, dst)].put(self._raw(array))
        return array

    def recv_like(self, template, src):
        return self._from_raw(self._box[(src, self._rank)].take(), tuple(template.shape), template.dtype)

    def all_gather(self, array):
        import mlx.core as mx

        raw = self._raw(array)
        self._box[("ag", self._rank)].put(raw)
        parts = []
        for r in range(self._size):
            if r == self._rank:
                parts.append(array)
            else:
                parts.append(self._from_raw(self._box[("ag", r)].take(), tuple(array.shape), array.dtype))
        # the writer must not overwrite before every peer consumed: put() waits for the consumed counter of the NEXT call
        return mx.concatenate(parts, axis=0)

    def flush_sends(self, sends):
        for array, dst in sends:
            self.send(array, dst)

    def step_complete(self):
        pass

    def barrier(self):
        self.all_gather(__import__("mlx.core").core.zeros((1,)))

    def any_true(self, value):
        import mlx.core as mx

        return bool(int(self.all_gather(mx.array([int(value)])).sum().item()) > 0)

    def close(self):
        for s in self._box.values():
            try:
                s.shm.close()
            except Exception:  # noqa: BLE001
                pass
        for shm in self._owned:
            try:
                if self._rank == 0 and shm is not None:
                    shm.unlink()
            except Exception:  # noqa: BLE001
                pass
