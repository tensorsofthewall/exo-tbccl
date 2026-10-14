# exo-tbccl

An optional package that lets [exo](https://github.com/exo-explore/exo) run pipeline-parallel text generation across machines of different kinds (for example a Mac with Metal and a Linux host with CUDA), with TBCCL moving the activations between stages.

```{note}
**Release candidate.** The current release candidate is 0.3.0rc2 (TBCCL C ABI 1 / wire protocol 4, MLX 0.32). It is a pre-release for validation, not for production use. The final 0.3.0 has not been released.
```

exo-tbccl is one of three independent adapters over the [TBCCL](https://tbccl.tensorsofthewall.com/en/latest/) runtime; the others are torch-tbccl (PyTorch) and vllm-tbccl (vLLM). It depends on neither. See the [ecosystem overview](project/ecosystem.md).

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/overview
getting-started/install
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

guides/two-host
reference/configuration
concepts/fast-paths
```

```{toctree}
:maxdepth: 2
:caption: Architecture

concepts/index
adr/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:maxdepth: 2
:caption: Troubleshooting

guides/troubleshooting
guides/discovery-troubleshooting
guides/verify-active
```

```{toctree}
:maxdepth: 2
:caption: Development

development/index
```

```{toctree}
:maxdepth: 2
:caption: Project

project/index
validation/index
```
