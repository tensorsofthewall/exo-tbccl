"""EXO_TBCCL_PREALLOC_GATHER: the next all_gather destination is prepared while a posted receive is in flight (Metal only, default off). Process-per-rank over loopback.
The setting is honoured on Metal only; the functional tests force the Metal gate so the path also runs on CUDA/CPU hosts."""

import pytest

from tests.harness import run_world
from tests.test_group import ADV

mx = pytest.importorskip("mlx.core")
METAL = bool(mx.metal.is_available())


def test_config_parses_the_setting_and_defaults_off():
    from exo_tbccl.config import FastPathConfig

    assert FastPathConfig().prealloc_gather is False
    assert FastPathConfig.from_env().prealloc_gather is False
    import os

    old = os.environ.get("EXO_TBCCL_PREALLOC_GATHER")
    try:
        os.environ["EXO_TBCCL_PREALLOC_GATHER"] = "1"
        assert FastPathConfig.from_env().prealloc_gather is True
        os.environ["EXO_TBCCL_PREALLOC_GATHER"] = "0"
        assert FastPathConfig.from_env().prealloc_gather is False
    finally:
        if old is None:
            os.environ.pop("EXO_TBCCL_PREALLOC_GATHER", None)
        else:
            os.environ["EXO_TBCCL_PREALLOC_GATHER"] = old


def _w_decode_steps(rank, world, ex, enabled, force_gate, shapes):
    import exo_tbccl.group as g
    from exo_tbccl.config import FastPathConfig
    from exo_tbccl.group import TbcclPipelineComm

    if force_gate:
        g._metal_available = lambda: True  # type: ignore[assignment]
    comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000, config=FastPathConfig(prealloc_gather=enabled))
    ok = True
    try:
        for step, n in enumerate(shapes):
            if rank == 0:
                act = mx.full((1, n), step + 1, dtype=mx.float32)
                mx.eval(act)
                comm.send(act, 1)
            else:
                got = comm.recv_like(mx.zeros((1, n), dtype=mx.float32), 0)
                ok = ok and bool(mx.all(got == step + 1).item())
            mine = mx.full((1, 4), 10 * rank + step, dtype=mx.float32)
            mx.eval(mine)
            out = comm.all_gather(mine)
            mx.eval(out)
            expect = mx.concatenate([mx.full((1, 4), step, dtype=mx.float32), mx.full((1, 4), 10 + step, dtype=mx.float32)])
            ok = ok and out.shape == (2, 4) and bool(mx.all(out == expect).item())
        comm.step_complete()
        return ok, comm.prealloc_used
    finally:
        comm.close()


def test_results_are_exact_and_the_prepared_destination_is_used():
    res = run_world(2, _w_decode_steps, True, True, [8] * 12)
    assert all(r[0] for r in res), res
    assert res[1][1] >= 10, res  # the receiving stage prepares during the receive and consumes at the next all_gather
    assert res[0][1] == 0, res  # the sending stage never posts a receive, so it never prepares


def test_a_changed_shape_discards_the_prepared_destination_and_stays_correct():
    res = run_world(2, _w_decode_steps, True, True, [8, 8, 8, 32, 32, 8])
    assert all(r[0] for r in res), res


def test_disabled_is_identical_and_prepares_nothing():  # negative control: without the setting the counter must stay at zero
    res = run_world(2, _w_decode_steps, False, True, [8] * 6)
    assert all(r[0] for r in res), res
    assert res[1][1] == 0 and res[0][1] == 0, res


def test_the_setting_is_inert_off_metal():
    res = run_world(2, _w_decode_steps, True, False, [8] * 6)
    assert all(r[0] for r in res), res
    if not METAL:
        assert res[1][1] == 0, res


def _w_close_with_prepared_destination(rank, world, ex):
    import exo_tbccl.group as g
    from exo_tbccl.config import FastPathConfig
    from exo_tbccl.group import TbcclPipelineComm

    g._metal_available = lambda: True  # type: ignore[assignment]
    comm = TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000, config=FastPathConfig(prealloc_gather=True))
    if rank == 0:
        for _ in range(2):
            a = mx.ones((1, 8), dtype=mx.float32)
            mx.eval(a)
            comm.send(a, 1)
            comm.all_gather(mx.ones((1, 4), dtype=mx.float32))
    else:
        for _ in range(2):
            comm.recv_like(mx.zeros((1, 8), dtype=mx.float32), 0)
            comm.all_gather(mx.ones((1, 4), dtype=mx.float32))
    prepared = comm._next_dest is not None
    comm.close()
    return prepared, comm._next_dest is None


def test_close_drops_a_prepared_destination():
    res = run_world(2, _w_close_with_prepared_destination)
    assert all(r[1] for r in res), res
