"""TbcclPipelineComm over loopback, one process per rank (spawn), exchange mediated by the parent like an application's all-gather."""

import pytest

from tests.harness import run_world

ADV = "127.0.0.1"


def _create(rank, world, ex):
    from exo_tbccl.group import TbcclPipelineComm

    return TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000)


def _w_host_p2p(rank, world, ex):
    import numpy as np

    comm = _create(rank, world, ex)
    try:
        assert (comm.rank(), comm.size()) == (rank, world)
        out = []
        for dtype, n in (("float32", 1024), ("float16", 333), ("uint8", 7), ("int32", 1)):
            for peer in range(world):
                if peer == rank:
                    continue
                payload = (np.arange(n) * (rank + 3) + peer).astype(dtype)
                got = np.zeros(n, dtype=dtype)
                if rank < peer:
                    comm.wait(comm.send_async(payload, peer))
                    comm.wait(comm.recv_into_async(got, peer))
                else:
                    comm.wait(comm.recv_into_async(got, peer))
                    comm.wait(comm.send_async(payload, peer))
                expect = (np.arange(n) * (peer + 3) + rank).astype(dtype)
                assert got.tobytes() == expect.tobytes(), (dtype, rank, peer)
                out.append((dtype, peer, True))
        comm.barrier()
        assert comm.any_true(rank == world - 1) is True
        assert comm.any_true(False) is False
        assert comm.stats.materialized_copies == 0 and comm.stats.staged_fallback_copies == 0
        return len(out)
    finally:
        comm.close()


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_host_send_recv_barrier_any_true(world):
    res = run_world(world, _w_host_p2p)
    assert all(r == 4 * (world - 1) for r in res)


def _w_mlx_roundtrip(rank, world, ex):
    import mlx.core as mx

    comm = _create(rank, world, ex)
    try:
        results = {}
        for dt in (mx.float32, mx.float16, mx.bfloat16):
            x = (mx.arange(2 * 3 * 5) * (rank + 1)).astype(dt).reshape(2, 3, 5)
            if world == 1:
                results[str(dt)] = True
                continue
            peer = 1 - rank
            if rank == 0:
                comm.send(x, peer)
                y = comm.recv_like(x, peer)
            else:
                y = comm.recv_like(x, peer)
                comm.send(x, peer)
            ref = (mx.arange(2 * 3 * 5) * (peer + 1)).astype(dt).reshape(2, 3, 5)
            results[str(dt)] = bool(mx.array_equal(y.view(mx.uint8), ref.view(mx.uint8)).item())
        # all_gather along axis 0, rank order
        x = mx.full((2, 3), rank + 1, dtype=mx.float32)
        g = comm.all_gather(x)
        expect = mx.concatenate([mx.full((2, 3), r + 1, dtype=mx.float32) for r in range(world)], axis=0)
        results["gather"] = bool(mx.array_equal(g, expect).item()) and g.shape == (2 * world, 3)
        # strided source is materialized (counted, correct)
        base = mx.arange(12, dtype=mx.float32).reshape(3, 4)
        t = base.T
        if world == 2:
            if rank == 0:
                comm.send(t, 1)
            else:
                y = comm.recv_like(t, 0)
                results["strided"] = bool(mx.array_equal(y, base.T).item())
        results["copies"] = comm.stats.materialized_copies
        return results
    finally:
        comm.close()


@pytest.mark.parametrize("world", [1, 2])
def test_mlx_arrays_roundtrip_bit_exact(world):
    res = run_world(world, _w_mlx_roundtrip)
    for r, d in enumerate(res):
        assert all(v is True for k, v in d.items() if k != "copies"), d
    if world == 2:
        assert res[0]["copies"] == 1 and res[1]["copies"] == 0


def _w_gather_n(rank, world, ex):
    import mlx.core as mx

    comm = _create(rank, world, ex)
    try:
        x = mx.full((1, 4), float(rank + 10), dtype=mx.bfloat16)
        g = comm.all_gather(x)
        ok = all(mx.array_equal(g[r], mx.full((4,), float(r + 10), dtype=mx.bfloat16)).item() for r in range(world))
        scalar = comm.all_gather(mx.array(rank, dtype=mx.int32))
        return ok and scalar.tolist() == list(range(world))
    finally:
        comm.close()


@pytest.mark.parametrize("world", [2, 3, 4])
def test_all_gather_rank_order_n234(world):
    assert all(run_world(world, _w_gather_n))
