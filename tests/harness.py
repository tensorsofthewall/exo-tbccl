"""Process-per-rank test harness: the parent plays the application's all-gather for the UniqueId / endpoint-blob exchange."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Callable, Sequence
from typing import Any

SPAWN = mp.get_context("spawn")


class Exchange:
    """The ByteExchange handed to TbcclPipelineComm.create inside a rank process."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def __call__(self, purpose: str, payload: bytes) -> Sequence[bytes]:
        self.conn.send(("xchg", purpose, payload))
        kind, value = self.conn.recv()
        if kind == "error":
            raise RuntimeError(value)
        return value


def _entry(rank: int, world: int, conn: Any, fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    try:
        result = fn(rank, world, Exchange(conn), *args)
        conn.send(("done", result))
    except BaseException as e:  # noqa: BLE001
        conn.send(("fail", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def run_world(world: int, fn: Callable[..., Any], *args: Any, timeout: float = 120.0, skip_rank_after_bootstrap: int | None = None) -> list[Any]:
    """Run fn(rank, world, exchange, *args) in `world` processes; returns each rank's result or raises with every rank's failure."""
    parents = []
    procs = []
    for r in range(world):
        a, b = SPAWN.Pipe()
        p = SPAWN.Process(target=_entry, args=(r, world, b, fn, args), daemon=True)
        p.start()
        b.close()
        parents.append(a)
        procs.append(p)
    results: list[Any] = [None] * world
    failures: dict[int, str] = {}
    finished = [False] * world
    rounds: dict[str, dict[int, bytes]] = {}
    import time

    deadline = time.time() + timeout
    while not all(finished) and time.time() < deadline:
        for r, conn in enumerate(parents):
            if finished[r] or not conn.poll(0.01):
                continue
            try:
                msg = conn.recv()
            except EOFError:
                finished[r] = True
                failures.setdefault(r, "rank process died without a result")
                continue
            if msg[0] == "xchg":
                _, purpose, payload = msg
                rounds.setdefault(purpose, {})[r] = payload
                if len(rounds[purpose]) == world:
                    gathered = [rounds[purpose][i] for i in range(world)]
                    for c in parents:
                        c.send(("ok", gathered))
                    del rounds[purpose]
            elif msg[0] == "done":
                results[r] = msg[1]
                finished[r] = True
            else:
                failures[r] = msg[1]
                finished[r] = True
        for r, p in enumerate(procs):
            if not finished[r] and not p.is_alive() and not parents[r].poll(0):
                finished[r] = True
                failures.setdefault(r, f"rank process exited with code {p.exitcode}")
    for r, p in enumerate(procs):
        if not finished[r]:
            failures.setdefault(r, "timed out")
    for p in procs:
        p.join(5)
        if p.is_alive():
            p.kill()
    if failures:
        raise AssertionError("\n".join(f"[rank {r}] {m}" for r, m in sorted(failures.items())))
    return results
