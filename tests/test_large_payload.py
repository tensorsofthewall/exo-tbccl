"""N=2 bf16 point-to-point and AllGather payloads at the sizes a 5120-wide pipeline (Qwen3.8-27B class) really moves: one decode activation
(1 x 5120), a prefill tail chunk (577 tokens) and a full exo prefill chunk (2048 tokens = 20,971,520 B). Bit-exact, with a send posted before the receive
(queued-prefill order) and the other way round; one communicator reused for all sizes, closed explicitly."""

import pytest

from tests.harness import run_world
from tests.test_group import ADV

mx = pytest.importorskip("mlx.core")

HIDDEN = 5120


def _pattern(rows, rank, salt):
    import mlx.core as mx_

    n = rows * HIDDEN
    # a deterministic, rank- and size-dependent bf16 pattern that exercises every byte of the payload
    v = (mx_.arange(n, dtype=mx_.float32) * 0.013 + rank * 7.0 + salt) % 251.0
    return v.astype(mx_.bfloat16).reshape(1, rows, HIDDEN)


def _w_large(rank, world, ex, sizes):
    import mlx.core as mx_

    from exo_tbccl.group import TbcclPipelineComm

    comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=60000)
    ok = {}
    try:
        for rows in sizes:
            x = _pattern(rows, rank, rows)
            mx_.eval(x)
            peer = 1 - rank
            ref = _pattern(rows, peer, rows)
            mx_.eval(ref)
            # queued send first (rank 0), receiver posts later; then the reverse direction
            if rank == 0:
                comm.send(x, peer)
                y = comm.recv_like(x, peer)
            else:
                y = comm.recv_like(x, peer)
                comm.send(x, peer)
            mx_.eval(y)
            ok[f"p2p_{rows}"] = bool(mx_.array_equal(y.view(mx_.uint16), ref.view(mx_.uint16)).item())
            g = comm.all_gather(x.reshape(rows, HIDDEN))
            mx_.eval(g)
            exp = mx_.concatenate([_pattern(rows, r, rows).reshape(rows, HIDDEN) for r in range(world)], axis=0)
            ok[f"gather_{rows}"] = bool(mx_.array_equal(g.view(mx_.uint16), exp.view(mx_.uint16)).item()) and tuple(g.shape) == (world * rows, HIDDEN)
        comm.barrier()
        ok["any_true"] = comm.any_true(rank == 1)
        ok["pending_end"] = len(comm._pending)
        ok["copies"] = comm.stats.materialized_copies
        return ok
    finally:
        comm.close()


@pytest.mark.parametrize("sizes", [(1, 577, 2048)])
def test_bf16_payloads_up_to_a_full_prefill_chunk_are_bit_exact(sizes):
    res = run_world(2, _w_large, sizes, timeout=300)
    for r, d in enumerate(res):
        assert all(v is True for k, v in d.items() if k.startswith(("p2p_", "gather_")) or k == "any_true"), d
        assert d["pending_end"] == 0 and d["copies"] == 0, d  # no hidden payload-sized copy, nothing outstanding


def _w_repeated(rank, world, ex, cycles):
    import gc

    import psutil

    from exo_tbccl.group import TbcclPipelineComm

    proc = psutil.Process()
    peer = 1 - rank
    base = None
    for i in range(cycles):
        comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=60000)
        x = _pattern(577, rank, i)
        if rank == 0:
            comm.send(x, peer)
            y = comm.recv_like(x, peer)
        else:
            y = comm.recv_like(x, peer)
            comm.send(x, peer)
        mx.eval(y)
        comm.barrier()
        comm.close()
        del comm, x, y
        gc.collect()
        if i == 1:
            base = (proc.num_fds(), proc.memory_info().rss)
    return proc.num_fds() - base[0], (proc.memory_info().rss - base[1]) / 1e6


def test_repeated_create_use_destroy_with_prefill_sized_payloads_leaks_nothing():
    for fds, rss_mb in run_world(2, _w_repeated, 8, timeout=600):
        assert fds == 0 and rss_mb < 150, (fds, rss_mb)
