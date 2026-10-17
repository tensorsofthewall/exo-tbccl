# Getting started

These commands run from a clone of this repository. The supported tuple is in `compatibility.json` and the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html): exo-tbccl 0.3.x with the maintained exo fork tag `tbccl-integration-v2`, MLX (CUDA 13 on Linux, Metal on macOS) and CPython 3.13. Install it as the [two-host guide](../guides/two-host.md) describes; this page does not change that environment. Models must already be on every machine; nothing here downloads one.

## Which version runs, and which version is the provider

Two things are called exo-tbccl here, and they can have different versions:

* **The runtime** is the installed `exo-tbccl` wheel in your Exo environment. The immutable `tbccl-integration-v2` tag of the Exo fork installs the **0.3.0** wheel through its `tbccl` extra and that stays so; 0.3.1 has the same adapter code, so installing the 0.3.1 wheel by hand is optional.
* **The benchmark provider** is the `benchmarks/provider` directory of the **0.3.1** source (the source archive of the tag `v0.3.1`). It is not part of any wheel.

So the normal setup is the 0.3.0 wheel running and the 0.3.1 provider directory configured as `exo_bench_dir`. The result records both: `provider_version` is the installed wheel (0.3.0 in that setup), and the hashes of the provider scripts identify the 0.3.1 source. Do not read `0.3.0` in a result as a statement about the provider scripts. The historical measurements on the results page used the 0.3.0 wheel.

## 1. Check what this environment can run

```sh
python benchmarks/provider/provider.py describe
```

One JSON object is printed; no model is loaded, no node is started and no network connection is opened. `availability.status` is `available`, `environment_missing` (with the missing package) or `unsupported`. Versions come from package metadata.

## 2. Measure a cluster you started

Start the Exo node on each machine the way the [two-host guide](../guides/two-host.md) says. On the API host, write a spec naming the API, the model, the backend you want and what you expect to be placed, and run the client:

```json
{
  "api": {"host": "<API_HOST>", "port": 52415},
  "model": "<MODEL_ID>",
  "backend": "MlxTbccl",
  "node_count": 2,
  "world_size": 2,
  "link_addresses": ["<LINK_ADDRESS_A>", "<LINK_ADDRESS_B>"],
  "prompts": {"medium": {"items": 3}},
  "phases": [
    {"name": "warm", "kind": "warmup", "prompt": "medium", "max_tokens": 32, "requests": 3},
    {"name": "lat", "kind": "rounds", "prompt": "medium", "max_tokens": 32, "conc": 1, "requests": 10}
  ]
}
```

```sh
python benchmarks/provider/client.py --spec spec.json --out-dir out
```

In order, the client waits until the expected nodes see each other, places one instance with Exo's normal placement (it never sets layer boundaries), reads back what was placed and **refuses to continue if it is not what you asked for** (instance type for the backend, world size, layer assignment if you gave `expected_layers`, link addresses), waits for every runner to finish loading the model (timed on its own), runs the phases, and deletes the instance. Every step prints a raw evidence line; every request writes its event timeline. A prompt is either `{"text": "..."}` or `{"items": N}`, a fixed text of N numbered sentences.

Set `backend` to `MlxRing` for the baseline. To compare backends, read [MlxRing versus MlxTbccl](ring-vs-tbccl.md) first.

## 3. Through the TBCCL benchmark runner

In a TBCCL checkout, name this provider per host (all paths and addresses are yours):

```yaml
hosts:
  host-a:
    executables:
      exo: <PATH_TO_EXO_COMMAND>
      exo_python: <PATH_TO_PYTHON_OF_THE_EXO_ENVIRONMENT>
      exo_bench_dir: <PATH_TO_THIS_REPOSITORY>/benchmarks/provider
```

```sh
python3 -m benchmarks.suite providers my-config.yaml          # shows exo-tbccl with its state and versions per host
python3 -m benchmarks.suite run my-config.yaml --dry-run      # prints the node, sampler and client commands; launches nothing
python3 -m benchmarks.suite run my-config.yaml --allow-physical
python3 -m benchmarks.suite report <RUN_ID>
python3 -m benchmarks.suite compare <RUN_A> <RUN_B>           # only equivalent experiments are compared
```

One workload point is one *cluster lifecycle*: nodes start, the client places an instance and runs its phases, everything is stopped and verified clean on every host, and the next lifecycle starts from scratch. Repetition means more lifecycles, not more requests.

## Tests

```sh
python -m unittest discover -s benchmarks/provider/tests -p 'test_*.py'
```

The tests run offline against a stand-in Exo API; they need no Exo, MLX, GPU or model.
