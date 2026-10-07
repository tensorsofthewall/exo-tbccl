# Related projects

exo-tbccl uses the TBCCL C ABI directly and does not use PyTorch. The projects below have their own documentation, version and release schedule.

| Project | Role | Source | Planned release |
|---|---|---|---|
| [tbccl](https://tbccl.tensorsofthewall.com/en/stable/) | The runtime: C++ library, stable C ABI, TCP transport and collectives. | [source](https://github.com/tensorsofthewall/tbccl) | 0.6.0 (unreleased) |
| [torch-tbccl](https://github.com/tensorsofthewall/torch-tbccl) | A PyTorch `torch.distributed` backend over an installed TBCCL. | [source](https://github.com/tensorsofthewall/torch-tbccl) | 0.2.0 (unreleased) |
| [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) | A vLLM platform plugin that routes communication through torch-tbccl. | [source](https://github.com/tensorsofthewall/vllm-tbccl) | 0.2.0 (unreleased) |

The core documentation is hosted at the stable site linked above once TBCCL's first release is tagged. The adapters' hosting is decided with their first releases; until then each project's documentation is built from its repository with `make docs` (see {doc}`development/building-docs`). The planned releases are unreleased targets, recorded in each project's `compatibility.json`.
