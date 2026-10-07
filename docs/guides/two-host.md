# Running across two hosts

Two-host pipeline-parallel generation uses exo's normal placement with the `MlxTbccl` instance type on a Linux CUDA host and a Mac, joined by a direct Thunderbolt 4 link. exo picks one address per node that every peer can reach (Thunderbolt preferred); TBCCL advertises one endpoint per rank, so per-peer addresses are rejected.

- **Link first.** Set up and check the link before starting ranks (see the [TBCCL Thunderbolt guide](https://github.com/tensorsofthewall/tbccl/blob/main/docs/guides/thunderbolt-link.md)): read the PCIe error counters before and after every real-link session and stop on any new fatal or nonfatal error.
- **Keep Mac ranks in a live session.** A background job outside a live SSH session or `tmux` cannot reach the local network on macOS and fails at bootstrap.
- **Memory placement.** exo advertises the host RAM of a CUDA node, not its GPU memory; on a small GPU set exo's memory override so the layer split fits.
- **Probes.** `examples/link_probe.py` is a two-host correctness probe, `examples/two_host_fastpath.py` and `examples/two_host_ring_chain.py` exercise the fast paths and the ring comparison, and `benchmarks/real_model_two_host.py` drives a real model; read each script's docstring for its arguments. Never download model weights for them.

Peer-failure behavior is validated on loopback; do not kill a physical peer to test it.
