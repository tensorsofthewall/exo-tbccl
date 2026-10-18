# Code map

Where to start reading, by topic. It is a map to the directories and the first files, not a walk through the source; the rules for the code base are in `AGENTS.md` and the design is in the [architecture](../concepts/architecture.md) page. Every path below is checked by a test, so a rename that leaves this page stale fails the documentation checks.

| Topic | Start at | Read with |
|---|---|---|
| The Exo integration boundary (what exo calls) | `exo_tbccl/__init__.py`, `exo_tbccl/group.py` | [Architecture](../concepts/architecture.md), [API](../reference/api.md) |
| The native binding over the C ABI | `src/native/exo_tbccl_native.c`, `exo_tbccl/_loader.py` | [Compatibility](../reference/compatibility.md) |
| The pipeline data plane (send, receive, gather, barrier) | `exo_tbccl/group.py`, `exo_tbccl/recv_pool.py` | [Bridge](../concepts/bridge.md), [lifetime and failure](../concepts/lifetime-and-failure.md) |
| MLX array to TBCCL buffer (DLPack) | `exo_tbccl/bridge.py`, `tests/test_dlpack_consumer.py` | [Bridge](../concepts/bridge.md) |
| Bootstrap: endpoints through exo's byte exchange | `exo_tbccl/group.py` | [Bootstrap](../concepts/bootstrap.md) |
| Discovery versus transport | exo's own discovery is not in this repository; the transport starts at bootstrap | [Discovery troubleshooting](../guides/discovery-troubleshooting.md), [verify](../guides/verify-active.md) |
| Typed errors | `exo_tbccl/errors.py` | [API](../reference/api.md) |
| Optional fast-path and activity settings (off by default) | `exo_tbccl/config.py`, `exo_tbccl/step_activity.py` | [Fast paths](../concepts/fast-paths.md), [Metal activity](../concepts/mac-activity.md), [configuration](../reference/configuration.md) |
| Docs and version gates | `tests/test_docs_consistency.py`, `compatibility.json`, `tools/check_framework_pin.py` | [Compatibility](../reference/compatibility.md) |
| Process-per-rank tests | `tests/test_group.py`, `tests/test_lifecycle.py`, `tests/harness.py` | [Testing](testing.md) |
| Two-host probes and benchmarks | `examples/link_probe.py`, `examples/two_host_fastpath.py`, `benchmarks/provider/` | [Benchmarks](../benchmarks/index.md) |
| Diagnostics | `tools/collect_diagnostics.py` | [Troubleshooting](../guides/troubleshooting.md) |

The runtime itself (transport, algorithms, protocol, the C ABI) is not in this repository; its [code map](https://tbccl.tensorsofthewall.com/en/latest/development/code-map.html) lives in TBCCL. Exo's own code (discovery, topology, placement, the runner) belongs to the exo project; this repository covers the integration seam and `MlxTbccl`'s communication ([where should this change go?](where-should-this-change-go.md)).
