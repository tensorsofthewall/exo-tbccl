"""Diagnostic-only capture of exo-tbccl's own trace inside Exo's runner processes (put this directory on PYTHONPATH of a traced node; nothing in Exo or exo-tbccl is modified).

Two things are missing from a normal run: exo-tbccl logs each pipeline operation on the stdlib logger ``exo_tbccl`` at INFO (``EXO_TBCCL_TRACE=1``), but Exo's runner configures no handler for it, so
the lines are dropped; and the adapter-level ``CopyStats`` (copies made by exo-tbccl itself, not staging done inside TBCCL's providers) live only in memory. With ``TBCCL_B6_TRACE_DIR`` set
this hook (1) attaches a file handler with wall-clock timestamps to ``exo_tbccl`` and (2) records every ``TbcclPipelineComm`` created in the process and rewrites a JSON snapshot of its
CopyStats every two seconds (a runner can be terminated without running exit handlers). With ``TBCCL_TRACE_TIMELINE=1`` it also records the per-step timeline (step_timeline.py). Without the variable it does nothing. Traced runs are never mixed with the performance runs.
"""
import os
import sys

_DIR = os.environ.get("TBCCL_B6_TRACE_DIR")


def _install():
    import atexit
    import dataclasses
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import json
    import logging
    import threading
    import time

    os.makedirs(_DIR, exist_ok=True)
    pid = os.getpid()
    handler = logging.FileHandler(os.path.join(_DIR, f"exo_tbccl-{pid}.log"))
    handler.setFormatter(logging.Formatter("%(created).6f %(message)s"))
    logger = logging.getLogger("exo_tbccl")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    comms = []
    lock = threading.Lock()

    def snapshot():
        with lock:
            items = list(comms)
        data = {"pid": pid, "written": time.time(), "comms": []}
        for c in items:
            try:
                data["comms"].append({"rank": getattr(c, "_rank", None), "stats": dataclasses.asdict(c.stats)})
            except Exception as e:  # noqa: BLE001
                data["comms"].append({"error": repr(e)})
        tmp = os.path.join(_DIR, f".copystats-{pid}.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(_DIR, f"copystats-{pid}.json"))

    def loop():
        while True:
            time.sleep(2.0)
            if comms:
                snapshot()

    def wrap(module):
        cls = module.TbcclPipelineComm
        original = cls.__init__

        def init(self, *a, **k):
            original(self, *a, **k)
            with lock:
                comms.append(self)
        cls.__init__ = init
        threading.Thread(target=loop, daemon=True, name="b6-copystats").start()
        atexit.register(lambda: comms and snapshot())

    callbacks = {"exo_tbccl.group": [lambda module: wrap(module)]}
    if os.environ.get("TBCCL_TRACE_TIMELINE", "0") not in ("", "0"):
        # Optional per-step timeline (trace_hook/step_timeline.py): no logging in the decode path, about 20 integer appends per step.
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import step_timeline
            recorder, after_import = step_timeline.install(_DIR)
            atexit.register(recorder.close)
            for target in ("exo.worker.engines.mlx.auto_parallel", "exo.worker.engines.mlx.pipeline_comm", "exo_tbccl.group"):
                callbacks.setdefault(target, []).append(lambda module, target=target: after_import(target, module))
        except Exception as e:  # noqa: BLE001 - a diagnostic must never stop the node it observes
            print(f"step timeline disabled: {e!r}", file=sys.stderr)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in callbacks:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)  # the parent package is already imported, so `path` is its __path__: nothing is imported here
            if spec is None or spec.loader is None:
                return spec
            real_exec = spec.loader.exec_module

            def exec_module(module):
                real_exec(module)
                for callback in callbacks[name]:
                    callback(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, Finder())


if _DIR:
    _install()
