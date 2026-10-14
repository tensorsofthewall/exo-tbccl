# Troubleshooting

Find the symptom, check the likely cause with the diagnostic, apply the fix. For nodes that do not find each other use [discovery troubleshooting](discovery-troubleshooting.md). To confirm TBCCL is carrying the pipeline use [verify](verify-active.md).

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `MlxTbccl requires the optional exo-tbccl package: exo_tbccl is not installed (No module named 'exo_tbccl')` at placement | the adapter is not in exo's environment: it was never installed, a plain `uv sync` removed a manually installed wheel (use `uv sync --inexact`), or the platform has no adapter wheel (only Linux x86-64 and macOS arm64 do, so the optional `tbccl` extra is empty elsewhere) | `.venv/bin/python -c "import exo_tbccl; print(exo_tbccl.is_available())"` | install the wheel (or the extra) on every node ([installing](../getting-started/install.md)) |
| placement error when selecting `MlxTbccl` | `exo_tbccl` does not import in exo's environment, or exo lacks the seam | `.venv/bin/python -c "import exo_tbccl; print(exo_tbccl.is_available())"` prints the reason | install the wheel into exo's environment; use the exo integration ([compatibility](../reference/compatibility.md)) |
| `MlxTbccl supports only Pipeline sharding` / `does not support image or CFG-parallel models` | the request asks for tensor parallelism or an image model | the request body | use `"sharding": "Pipeline"` and a text model |
| `MlxTbccl requires every node to be reachable from every other node` / one address per node | the topology needs different addresses for different peers | exo's topology | connect the nodes over one shared network (the Thunderbolt link) |
| the placement uses one node only | `min_nodes` not set, or the second node is not in the cluster | `/state` shows the nodes | set `"min_nodes": 2` and fix discovery |
| bootstrap times out (120 s) | a rank did not publish its endpoint, or its address is unreachable | the runner logs of both nodes; `ping` the address exo chose | check every runner started; check the link; on the Mac use a live session |
| `TbcclProtocolMismatchError` or `protocol_mismatch ... wire protocol` | the two hosts have exo-tbccl wheels bundling different TBCCL wire protocols | `exo_tbccl.__version__` on both | install the same exo-tbccl version on both |
| `import exo_tbccl` fails after a dependency sync | `uv sync` in exo removed the package | `pip list \| grep exo` | reinstall the wheel |
| the exo process does not exit after an `MlxTbccl` instance | the runner byte-exchange inbox was not drained (fixed in the exo integration; older branches lack the fix) | the exo commit | use the exo version recorded in [compatibility](../reference/compatibility.md) |
| a send error appears later than the send | `EXO_TBCCL_ASYNC_SEND=1` holds a failed send until the next communication point | expected | the error is raised no later than the next synchronizing call |
| the model is slow on the first request | cold start of the model on each node | time the second request | warm up before measuring |
| a CUDA node runs out of memory while loading | exo advertises host RAM, not GPU memory | `nvidia-smi` | set `OVERRIDE_MEMORY_MB` for the node so the layer split fits |

A peer that dies surfaces as a structured transport error on the survivors within about a second; the runner fails and exo reports an error to the client. Cancellation never fails the instance. Peer-failure behavior is validated on loopback.
