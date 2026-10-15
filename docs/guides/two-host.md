# Running across two hosts

Two-host pipeline-parallel generation uses exo's normal placement with the `MlxTbccl` instance type on a Linux CUDA host and a Mac, joined by a direct Thunderbolt 4 link. exo picks one address per node that every peer can reach (Thunderbolt preferred); TBCCL advertises one endpoint per rank, so per-peer addresses are rejected.

## Before you start

1. **A verified link.** If the two machines have never been connected, follow [Two machines over Thunderbolt](https://tbccl.tensorsofthewall.com/en/latest/getting-started/two-machine-thunderbolt.html) first: addresses, ping **and** TCP in both directions. Interface names (`thunderbolt0`, `bridge0`) and addresses in the validated runs are specific to that pair; read yours from the machines.
2. **IPv6 link-local on the link.** A PyTorch or vLLM collective needs only IPv4. **Exo also needs IPv6 link-local multicast on the same interface**, so a successful all-reduce does not prove that exo will find the other machine. Check `ip -6 addr show <interface>` (Linux) and `ifconfig <bridge> | grep inet6` (Mac) each show an `fe80::` address, and see [discovery troubleshooting](discovery-troubleshooting.md) if not.
3. **exo and exo-tbccl installed on both machines**, from the immutable tag `tbccl-integration-v2` with the `tbccl` extra ([installing](../getting-started/install.md)). Use only the commands on that page; nothing else is needed to install the adapter.
4. **The model, already on both machines.** exo-tbccl never downloads weights, and with `EXO_OFFLINE=true` exo does not either. exo looks for a model in its models directory, which is `$EXO_HOME/models` (or `~/.local/share/exo/models` if `EXO_HOME` is unset; `EXO_DEFAULT_MODELS_DIR` and `EXO_MODELS_READ_ONLY_DIRS` add others), in a directory named after the model id with `/` replaced by `--`, for example `mlx-community--Qwen3-0.6B-8bit`. Copy the same model directory to both machines yourself.
5. **Ports.** The ports below must be reachable between the two link addresses; on Linux allow them from the peer's address on the link interface only, and on the Mac allow Python when prompted.

## The validated launch procedure

These were the conditions of the validated runs. Items marked *workaround* are operational findings from the validated environment, not architectural requirements of Thunderbolt or of exo-tbccl.

1. **Check the link first.** `ping` both directions; read the PCIe error counters before and after every real-link session and stop on any new fatal or nonfatal error ([TBCCL Thunderbolt guide](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-link.html)). Keep IPv6 enabled on the link interface: exo's discovery uses IPv6 link-local multicast, and both ends of the validated link had IPv6 link-local addresses (Linux `thunderbolt0`, Mac `bridge0`).
2. **Same namespace on every node.** `EXO_ZENOH_NAMESPACE=<name>` (or `--namespace`).
3. **Start the Linux node first, then the Mac a few seconds later.** Discovery works in either order, but this is the order that was validated.
4. **Workaround: launch the Mac node from an interactive Terminal window on the Mac or from a live `ssh -tt` session, not from a detached tmux session or a `nohup` job.** If you are sitting at both machines, just open a Terminal on each and run exo there; no SSH is needed. In the validated environment, a Mac exo node started from a detached tmux session failed to send its discovery packets (every send failed with `No route to host`), and exo's discovery then stopped announcing for the rest of the process's life, so the two nodes never formed a cluster. The same node started from a live `ssh -tt` session worked every time (three complete cycles). macOS also blocks background jobs without a live session from reaching the local network ([TBCCL link guide](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-link.html)).

   ```sh
   ssh -tt <mac> 'export PATH=/opt/homebrew/bin:$PATH EXO_HOME=$HOME/.exo EXO_ZENOH_NAMESPACE=mycluster; cd <exo checkout> && exec .venv/bin/exo'
   ```

5. **Workaround: do not run another program that listens on UDP port 52413 on the same host** (for example a packet-capture or debugging listener). On macOS such a listener can take the unicast discovery replies away from exo and stop the nodes from discovering each other.
6. **Allow the firewall prompt once.** The first time exo's Python listens on the Mac, macOS asks for permission; answer it from a local session. Do not script firewall changes.
7. **Memory placement.** exo advertises the host RAM of a CUDA node, not its GPU memory; on a small GPU set exo's memory override (`OVERRIDE_MEMORY_MB`) so the layer split fits.

## What "ready" looks like

- On each machine the exo log shows its own start and, within seconds of the second node starting, a `discovered` line for the other node's id; exactly one node later logs `elected master - demoting self`.
- `curl -s localhost:52415/state` (see the [quickstart](../getting-started/quickstart.md)) lists **2 nodes** on both machines.
- After placement, every runner reports ready before you send a request.

## Shutting down

Delete the instance (`DELETE /instance/<id>`), then stop each exo with Ctrl-C in its own terminal. Check `lsof -nP -iTCP:52414 -sTCP:LISTEN` and `lsof -nP -iUDP:52413` show nothing afterwards. Do not kill by process-name pattern.

## Ports

| Port | Protocol | Used for |
|---|---|---|
| 52413 | UDP, IPv6 multicast group `ff12::e0a1:de89` | exo node discovery |
| 52414 | TCP | exo's control plane (zenoh) |
| 52415 | TCP | exo's HTTP API |
| chosen per communicator | TCP | TBCCL control and data connections between the stages |

## Probes

`examples/link_probe.py` is a two-host correctness probe, `examples/two_host_fastpath.py` and `examples/two_host_ring_chain.py` exercise the fast paths and the ring comparison, and `benchmarks/real_model_two_host.py` drives a real model; read each script's docstring for its arguments. Never download model weights for them.

Peer-failure behavior is validated on loopback; do not kill a physical peer to test it.
