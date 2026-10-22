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


def _gate_and_create(rank, world, ex, reuse):
    import exo_tbccl.group as g
    from exo_tbccl.config import FastPathConfig
    from exo_tbccl.group import TbcclPipelineComm

    g._metal_available = lambda: True  # type: ignore[assignment]
    return TbcclPipelineComm.create(rank, world, ex, bind_host=ADV, advertise_host=ADV, timeout_ms=20000, config=FastPathConfig(prealloc_gather=True, recv_reuse=reuse))


def _w_close_with_prepared_destination(rank, world, ex, how):
    """Rank 1 ends with a prepared destination nobody consumed (a final recv_like with no all_gather after it), then tears down in the way under test."""
    import gc
    import time
    import weakref

    comm = _gate_and_create(rank, world, ex, False)
    n = 3
    if rank == 0:
        for _ in range(n):
            a = mx.ones((1, 8), dtype=mx.float32)
            mx.eval(a)
            comm.send(a, 1)
            if _ < n - 1:
                comm.all_gather(mx.ones((1, 4), dtype=mx.float32))
        time.sleep(1.0)
        try:
            comm.close()
        except Exception:  # noqa: BLE001
            pass
        return None
    for _ in range(n):
        comm.recv_like(mx.zeros((1, 8), dtype=mx.float32), 0)
        if _ < n - 1:
            comm.all_gather(mx.ones((1, 4), dtype=mx.float32))
    existed = comm._next_dest is not None
    ref = weakref.ref(comm._next_dest) if existed else None
    err = None
    if how == "abort":
        comm.abort("test abort with a prepared destination")
    if how == "close_raises":
        comm._reap_for_close = lambda: (_ for _ in ()).throw(RuntimeError("injected teardown failure"))  # type: ignore[assignment]
    try:
        comm.close()
    except RuntimeError as e:
        err = str(e)
    gc.collect()
    return existed, comm._next_dest is None, ref is not None and ref() is None, err


@pytest.mark.parametrize("how", ["close", "abort", "close_raises"])
def test_close_releases_a_prepared_destination_that_really_exists(how):
    res = run_world(2, _w_close_with_prepared_destination, how)
    existed, released, collected, err = res[1]
    assert existed, "the scenario must leave an unused prepared destination for close() to release"
    assert released and collected, res
    assert (err is not None) == (how == "close_raises"), res


def _w_prepare_fails(rank, world, ex, reuse, abort_fails):
    """Rank 1's destination preparation raises after the receive is posted. Rank 0 sends the activation unless the scenario wants the receive to stay outstanding."""
    import time

    import exo_tbccl.group as g

    comm = _gate_and_create(rank, world, ex, reuse)
    if rank == 0:
        if not abort_fails:
            a = mx.ones((1, 8), dtype=mx.float32)
            mx.eval(a)
            try:
                comm.send(a, 1)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(2.0)
        try:
            comm.close()
        except Exception:  # noqa: BLE001
            pass
        return None
    g._ABANDON_DRAIN_MS = 500  # keep the not-terminal scenario quick
    comm._gather_key = ((2, 4), mx.float32)  # so the preparation would run

    def boom():
        raise ValueError("injected preparation failure")

    comm._prepare_gather_dest = boom  # type: ignore[assignment]
    if abort_fails:
        def abort_boom(reason):
            raise OSError("injected abort failure")

        comm._abort_for_abandon = abort_boom  # type: ignore[assignment]
    out = {}
    try:
        comm.recv_like(mx.zeros((1, 8), dtype=mx.float32), 0)
        out["raised"] = None
    except BaseException as e:  # noqa: BLE001
        out["raised"] = type(e).__name__
        out["notes"] = list(getattr(e, "__notes__", []))
    out["pending_after"] = len(comm._pending)
    out["slots_after"] = comm.pool.slot_count
    out["aborted"] = bool(comm.is_aborted())
    t0 = time.time()
    try:
        comm.close()
        out["close_error"] = None
    except Exception as e:  # noqa: BLE001
        out["close_error"] = repr(e)
    out["close_s"] = time.time() - t0
    out["pending_closed"] = len(comm._pending)
    return out


@pytest.mark.parametrize("reuse", [False, True], ids=["fresh_destination", "reused_destination"])
def test_a_failed_preparation_drains_the_posted_receive_and_propagates_the_original_error(reuse):
    out = run_world(2, _w_prepare_fails, reuse, False)[1]
    assert out["raised"] == "ValueError", out
    assert out["pending_after"] == 0, out  # the receive is terminal and its borrow released
    assert out["slots_after"] == 0, out  # a leased pool slot (reuse) is discarded, never reused
    assert out["aborted"], out
    assert out["close_error"] is None and out["close_s"] < 10, out
    assert not out["notes"], out  # clean cleanup attaches nothing


@pytest.mark.parametrize("reuse", [False, True], ids=["fresh_destination", "reused_destination"])
def test_a_failing_cleanup_does_not_mask_the_original_error_and_teardown_stays_bounded(reuse):
    out = run_world(2, _w_prepare_fails, reuse, True)[1]
    assert out["raised"] == "ValueError", out  # the abort failure is not raised in its place
    assert any("abort failed" in n for n in out["notes"]), out
    assert any("not terminal" in n for n in out["notes"]), out  # nobody sent, the abort failed: the receive is still owned
    assert out["pending_after"] == 1, out  # its borrow stays alive until close()
    assert out["close_error"] is None and out["close_s"] < 15, out
    assert out["pending_closed"] == 0, out
