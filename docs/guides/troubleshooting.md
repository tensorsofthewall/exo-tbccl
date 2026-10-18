# Troubleshooting

Find the symptom, check the likely cause with the diagnostic, apply the fix. For nodes that do not find each other use [discovery troubleshooting](discovery-troubleshooting.md). To confirm TBCCL is carrying the pipeline use [verify](verify-active.md).

## Start here

### Work through it in this order

Each step is marked **Observe** (it reads state; nothing starts, stops or changes) or **Action** (you start, stop or configure a process, or run a test). No step changes network, firewall, driver or hardware settings. Stop at the first step that fails: later steps depend on it. Discovery (Exo finding the other machine) and transport (TBCCL moving the activations) are different layers; do not skip from one to the other.

```{table}
:class: tb-stack tb-stack-steps

| Step | Question | Check | If it fails |
|---|---|---|---|
| 1 | Which integration and versions? | **Observe:** `git -C <exo-fork checkout> describe --tags --exact-match`; `.venv/bin/python -c "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())"` on **every** node | [Symptoms](#symptoms-and-fixes); [installing](../getting-started/install.md) |
| 2 | Is the device and backend there? | **Observe:** `is_available()` prints `(True, '')`; MLX runs on the node | [Symptoms](#symptoms-and-fixes) |
| 3 | Is the link ready? | **Observe:** `python3 thunderbolt_link_doctor.py --peer <peer-link-address>` (with `--peer` it also sends a ping and opens one TCP connection to the peer, and changes nothing; [how to get it](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html)) | [Thunderbolt link troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html) |
| 4 | Do the nodes discover each other? | **Observe:** `/state` shows `2 nodes` | [Discovery troubleshooting](discovery-troubleshooting.md) |
| 5 | Is `MlxTbccl` placed? | **Observe:** `/state` lists `MlxTbcclInstance`, layers on both nodes | [Symptoms](#symptoms-and-fixes); [verify](verify-active.md) |
| 6 | Does the TBCCL bootstrap finish? | **Observe:** each node logs `TBCCL communicator ready` within the 120 s bootstrap | [Ping works but exo or TBCCL does not connect](#ping-works-but-exo-or-tbccl-does-not-connect) |
| 7 | Did stopping leave a process? | **Action:** stop each node with Ctrl-C; then **Observe:** `ps` for stale `exo` processes | [Symptoms](#symptoms-and-fixes) (the exit rows) |
| 8 | Is it only slow? | **Observe:** the [benchmark methodology](../benchmarks/methodology.md) | [Results](../benchmarks/results.md) |
```

### Find your symptom

```{table}
:class: tb-stack tb-stack-find

| You see | Go to |
|---|---|
| `MlxTbccl requires the optional exo-tbccl package` (the adapter or backend is not active) | [Symptoms](#symptoms-and-fixes), first row; [installing](../getting-started/install.md) |
| the nodes never see each other, or a peer expires after ten seconds | [Discovery troubleshooting](discovery-troubleshooting.md) |
| ping works but nothing connects | [Ping works but exo or TBCCL does not connect](#ping-works-but-exo-or-tbccl-does-not-connect) |
| `protocol_mismatch ... wire protocol` (`TbcclProtocolMismatchError`) | [Symptoms](#symptoms-and-fixes); install the same exo-tbccl version everywhere |
| bootstrap times out (120 s) | [Ping works but exo or TBCCL does not connect](#ping-works-but-exo-or-tbccl-does-not-connect) |
| the placement uses one node | [Symptoms](#symptoms-and-fixes) (`min_nodes`) |
| a CUDA node runs out of memory while loading | [Symptoms](#symptoms-and-fixes) (`OVERRIDE_MEMORY_MB`) |
| the exo process does not exit | [Symptoms](#symptoms-and-fixes) |
| slower than `MlxRing` | not a fault: [MlxRing versus MlxTbccl](../benchmarks/ring-vs-tbccl.md) and the [results](../benchmarks/results.md) |
```

### Which layer owns the error

```{table}
:class: tb-stack tb-stack-owner

| Layer | What it looks like | What to do |
|---|---|---|
| exo-tbccl (this adapter) | a placement error that names `exo_tbccl`, or a typed `Tbccl...Error` | this guide; report with the `exo_tbccl` version and `is_available()` |
| Native TBCCL | a result tag (`protocol_mismatch`, `timeout`) inside the message | [TBCCL troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/troubleshooting.html) |
| Exo | discovery, topology, placement, namespace, master election, the runner byte exchange | the [discovery guide](discovery-troubleshooting.md); upstream exo for anything else |
| MLX | array or device behavior, the MLX build | upstream MLX |
| Operating system | macOS Local Network prompt or firewall, a detached session, a port in use | [link troubleshooting entries 9 and 10](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html) |
| Network configuration | no IPv6 link-local address, wrong subnet, traffic over Wi-Fi | [link troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html), entries 5 to 8 and 11 |
| External dependencies | `uv sync`, the dashboard build, Xcode and the Metal toolchain | their own notes |
```

## Symptoms and fixes

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

## Ping works but exo or TBCCL does not connect

**Symptom.** `ping` answers, but the nodes do not form a cluster, or the cluster forms and the TBCCL bootstrap times out (120 s).

**Which layer.** `ping` proves an IPv4 ICMP path. Exo's discovery needs IPv6 link-local multicast (UDP 52413) and then a TCP connection (52414); TBCCL then needs IPv4 TCP connections between the runners' own ports. Any one can fail while ping works.

**Checks.** The ping and the log line are observations. The connection test is an **action**: it opens one TCP connection and closes it at once, which the listening side may log as a failed handshake, so run it only when no job you care about is starting. The link doctor's listener on a spare port (`--listen` / `--tcp-port`) tests the same path without touching a running job. Run them while the listening side is waiting.

```sh
# 1. observe (sends three ping packets) - discovery: scoped IPv6 reachability, both directions (Linux; on macOS use ping6 and bridge0, see discovery troubleshooting)
ping -6 -c3 'fe80::PEER-ADDRESS%thunderbolt0'
# 2. ACTION (opens and closes one TCP connection) - Exo's TCP port on the other node (run from the other machine while exo is running there)
python3 -c "import socket; socket.create_connection(('<peer-link-address>', 52414), timeout=3).close(); print('TCP connect ok')"
# 3. observe - is the bootstrap address the link? Read the log line "TBCCL bootstrap on <address>" in each node's exo log
```

- Step 1 fails: discovery cannot work. [Discovery troubleshooting](discovery-troubleshooting.md), question 6, and entry 11 of the [link guide](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html).
- Step 2 `timed out`: a firewall or wrong address; `Connection refused`: exo is not running there. 
- The cluster forms and the bootstrap times out: `TBCCL bootstrap on <address>` in each node's log must be the link address; a different address means the runners advertised another interface. Check the link doctor's TCP mode (entry 8).

On macOS start exo from a live Terminal or `ssh -tt` session and answer the prompt. **Do not** change firewall rules, MTU or interface settings as a first step.


## When to ask for help

Escalate when the checks isolate the layer and it is exo-tbccl or TBCCL, or when you cannot isolate it. Include the exo-fork tag, `exo_tbccl.__version__` and `is_available()` from every node, each node's exo log from start-up, the output of the discovery questions, `python tools/collect_diagnostics.py` output (versions and interface names only; nothing is uploaded) and the commands you ran. See [getting help](https://tbccl.tensorsofthewall.com/en/latest/project/getting-help.html). Exo's own behavior belongs to the exo project. Security problems follow the [security policy](../project/security-policy.md).
