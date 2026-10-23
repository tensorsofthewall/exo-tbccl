# exo-tbccl

:::{container} tb-lead
exo-tbccl is an optional package that lets [exo](https://github.com/exo-explore/exo) run pipeline-parallel text generation across machines of different kinds, for example a Mac with Metal and a Linux host with CUDA, with TBCCL moving the activations between stages. exo still owns discovery, placement and scheduling; exo-tbccl only moves the bytes.
:::

:::{container} tb-actions
[Get started](getting-started/quickstart.md) [Two-host guide](guides/two-host.md)
:::

```{note}
**Release.** The current release is 0.5.0 (TBCCL C ABI 1 / wire protocol 4, MLX 0.32), which bundles TBCCL 0.7.1 and adds two opt-in Metal options. It is experimental.
```

<nav class="tb-eco" aria-label="TBCCL ecosystem: the four documentation sites">
<ul>
<li class="tb-tbccl"><a href="https://tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">tbccl</span><span class="tb-eco-role">native runtime</span></a></li>
<li class="tb-torch"><a href="https://torch-tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">torch-tbccl</span><span class="tb-eco-role">PyTorch backend</span></a></li>
<li class="tb-vllm"><a href="https://vllm-tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">vllm-tbccl</span><span class="tb-eco-role">vLLM integration</span></a></li>
<li class="tb-exo" aria-current="page"><span><span class="tb-eco-name">exo-tbccl</span><span class="tb-eco-role">exo integration</span><span class="tb-eco-here">This site</span></span></li>
</ul>
</nav>

## What you get

::::{container} tb-cards
:::{container} tb-card tb-exo

<h3>Pipeline communication for exo</h3>

Adds an exo instance type, `MlxTbccl`, so the stages of a pipeline exchange activations over TBCCL's stable C ABI.

[Overview](getting-started/overview.md) · [Fast paths](concepts/fast-paths.md)
:::

:::{container} tb-card tb-exo

<h3>Mixed hosts</h3>

A Mac with Metal and a Linux host with CUDA, over a Thunderbolt 4 link or any TCP network.

[Two-host guide](guides/two-host.md)
:::

:::{container} tb-card tb-exo

<h3>Backend selection</h3>

Choose it per placement with `"instance_meta": "MlxTbccl"` and `"sharding": "Pipeline"`. It is offered only when `exo_tbccl` imports in exo's environment.

[Overview](getting-started/overview.md) · [Verify it is active](guides/verify-active.md)
:::

:::{container} tb-card tb-exo

<h3>Built on TBCCL</h3>

The wheel carries a statically linked libtbccl and does not use PyTorch.

[TBCCL documentation](https://tbccl.tensorsofthewall.com/en/latest/) · [GitHub](https://github.com/tensorsofthewall/exo-tbccl)
:::
::::

## Get started

::::{container} tb-cards
:::{container} tb-card 

<h3>Install</h3>

The wheel, and the exo fork the integration needs.

[Installing exo-tbccl](getting-started/install.md)
:::

:::{container} tb-card 

<h3>Quickstart</h3>

From installation to a first generation on a pipeline.

[Quickstart](getting-started/quickstart.md)
:::

:::{container} tb-card 

<h3>Run on two hosts</h3>

A Mac and a Linux CUDA host in one exo cluster.

[Two-host guide](guides/two-host.md)
:::

:::{container} tb-card 

<h3>Check it is active</h3>

Confirm that the cluster really runs over `MlxTbccl`.

[Verify it is active](guides/verify-active.md)
:::

:::{container} tb-card 

<h3>Tune the data path</h3>

The fast paths and what they cost.

[Fast paths](concepts/fast-paths.md)
:::
::::

## What is supported

Only configurations that were actually exercised are listed as validated. Everything else is not supported or not claimed.

::::{container} tb-cols
:::{container}
**Validated**

- A real exo cluster of a Linux CUDA host and a Mac with Metal over a direct Thunderbolt 4 link, with Qwen3-0.6B (three full start-to-shutdown cycles) and a 27B 4-bit Qwen3 model (short prompts, both orientations).
- A standalone transport probe between a CUDA rank and a Metal rank, bit for bit.
- Two-rank loopback pipelines whose token ids matched the unsharded greedy reference.
:::
:::{container}
**Not supported or not claimed**

- Pipeline parallelism for text generation only: tensor parallelism and image models are rejected.
- The integration is not in upstream exo; it needs the exo fork tag `tbccl-integration-v3`.
- Two stages were validated on hardware; models other than Qwen3, and larger prompts on an 8 GB GPU, are not claimed.
- One advertised address per node; a node started from a detached tmux session on a Mac did not join the cluster.
:::
::::

**Python.** The package metadata requires CPython 3.13 or newer (`requires-python`). CPython 3.13 is the only interpreter with a validated prebuilt wheel: the published wheels are `cp313` only (Linux x86-64 and macOS 14 or newer on Apple silicon), so any other interpreter would need a source build, which is not validated. Details: [compatibility](reference/compatibility.md) and [installing exo-tbccl](getting-started/install.md).

Authoritative pages: [compatibility](reference/compatibility.md), [validated configurations](project/validated-configurations.md) and [known limitations](project/limitations.md).

## Where it fits

exo-tbccl is one of three independent adapters over the [TBCCL](https://tbccl.tensorsofthewall.com/en/latest/) runtime; the others are torch-tbccl (PyTorch) and vllm-tbccl (vLLM). It depends on neither. See the [ecosystem overview](project/ecosystem.md).

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started

getting-started/overview
getting-started/install
getting-started/quickstart
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: User guide

guides/two-host
reference/configuration
concepts/fast-paths
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Benchmarks

benchmarks/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Architecture

concepts/index
adr/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Troubleshooting

guides/troubleshooting
guides/discovery-troubleshooting
guides/verify-active
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Development

development/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Project

project/index
validation/index
```
