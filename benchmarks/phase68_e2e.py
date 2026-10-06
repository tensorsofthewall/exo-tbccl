"""Phase 68: drive the REAL exo runtime (master + API + placement + workers + runners) over two nodes through its HTTP API. Nothing in exo is modified.

    python benchmarks/phase68_e2e.py state                                   # nodes, advertised memory, backends
    python benchmarks/phase68_e2e.py preview <model_id>                      # exo's own placement previews (MlxRing / Pipeline and Tensor)
    python benchmarks/phase68_e2e.py run <label> <MlxRing|MlxTbccl> <model_id> --prompts short:8,medium:32 [--reps 1] [--out FILE] [--keep]

`run` places the instance with exo's normal placement (Sharding.Pipeline, min_nodes 2, no manual layer boundaries), waits for every runner to be ready (the model load),
streams deterministic (temperature 0) chat completions and records, per prompt: text, token strings, usage, TTFT, per-token arrival times (decode intervals: median,
p25, p75, p95), the finish reason and any error; then deletes the instance. The prompt prefill is TTFT minus one decode interval (the stream carries no separate prefill event).
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

API = "http://localhost:52415"


def call(method, path, body=None, timeout=900):
    req = urllib.request.Request(API + path, method=method, data=None if body is None else json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def state():
    return call("GET", "/state")


def prompts():
    short = "Count from one to five in words."
    # Phase 68: "medium" = 6 items (~140 prompt tokens): the largest that the 8 GB GPU prefills with the exo-selected 21-layer shard (12 items ~266 tokens and the Phase 53
    # 17-item ~375-token prompt OOM in mlx-lm's CUDA gated-delta fallback, ~3.4 MB per token per linear layer; see docs/phase68_large_model_loading.md)
    medium = "Summarize the following in two sentences. " + " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog number {i * 3}." for i in range(1, 7))
    medium12 = "Summarize the following in two sentences. " + " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog number {i * 3}." for i in range(1, 13))
    medium17 = "Summarize the following in two sentences. " + " ".join(f"Item {i}: the quick brown fox number {i} jumps over lazy dog number {i * 3}." for i in range(1, 18))
    long = "Read the log and report the last line number. " + " ".join(f"Line {i}: alpha beta gamma delta epsilon {i % 17}." for i in range(1, 520))
    return {"short": short, "medium": medium, "medium12": medium12, "medium17": medium17, "long": long}


def place(meta, model, min_nodes=2):
    before = set(state()["instances"])
    t0 = time.time()
    call("POST", "/place_instance", {"model_id": model, "sharding": "Pipeline", "instance_meta": meta, "min_nodes": min_nodes})
    for _ in range(200):
        new = set(state()["instances"]) - before
        if new:
            iid = new.pop()
            return iid, state()["instances"][iid], time.time() - t0
        time.sleep(0.3)
    raise RuntimeError("no instance appeared")


def layout_of(inst):
    kind = next(iter(inst))
    inner = inst[kind]
    n2r = inner["shardAssignments"]["nodeToRunner"]
    r2n = {v: k for k, v in n2r.items()}
    shards = []
    for rn, s in inner["shardAssignments"]["runnerToShard"].items():
        m = s["PipelineShardMetadata"]
        shards.append({"device_rank": m["deviceRank"], "start_layer": m["startLayer"], "end_layer": m["endLayer"], "n_layers": m["endLayer"] - m["startLayer"],
                       "world_size": m["worldSize"], "node": r2n[rn], "runner": rn})
    return kind, sorted(shards, key=lambda s: s["device_rank"])


def wait_ready(iid, timeout=1800):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        st = state()
        inst = st["instances"].get(iid)
        if inst is None:
            raise RuntimeError("instance vanished")
        inner = next(iter(inst.values()))
        runners = inner["shardAssignments"]["runnerToShard"].keys()
        tags = {r: list(st["runners"].get(r, {"?": 0}).keys())[0] for r in runners}
        if tags != last:
            print(f"  [{time.time() - t0:6.1f}s] runners: {sorted(tags.values())}", flush=True)
            last = tags
        if all(t == "RunnerReady" for t in tags.values()):
            return time.time() - t0, tags
        if any("Fail" in t for t in tags.values()):
            raise RuntimeError(f"runner failed: {st['runners']}")
        time.sleep(1)
    raise TimeoutError(f"not ready: {last}")


def delete(iid):
    call("DELETE", f"/instance/{iid}")
    t0 = time.time()
    for _ in range(300):
        if iid not in state()["instances"]:
            return time.time() - t0
        time.sleep(0.3)
    raise RuntimeError("instance not deleted")


def chat_stream(model, prompt, max_tokens):
    t0 = time.time()
    req = urllib.request.Request(API + "/v1/chat/completions", method="POST", headers={"content-type": "application/json"},
                                 data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens, "stream": True,
                                                  "logprobs": True, "top_logprobs": 5, "enable_thinking": False}).encode())
    text, toks, times, usage, finish, error = "", [], [], None, None, None
    lps = []  # per generated token: chosen logprob and the top-5 alternatives (token, logprob)
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("error"):
                error = ev["error"]
            for ch in ev.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                c = (ch.get("delta") or {}).get("content")
                if c:
                    text += c
                    times.append(time.time() - t0)
                lp = (ch.get("logprobs") or {}).get("content") or []
                toks += [x["token"] for x in lp]
                lps += [{"token": x["token"], "logprob": x.get("logprob"), "top": [(t.get("token"), t.get("logprob")) for t in (x.get("top_logprobs") or [])]} for x in lp]
    gaps = [b - a for a, b in zip(times, times[1:])]
    q = lambda p: sorted(gaps)[min(len(gaps) - 1, int(p * len(gaps)))] if gaps else None
    return {"text": text, "tokens": toks, "logprobs": lps[:8], "ttft": times[0] if times else None, "token_times": times, "tpot_median": statistics.median(gaps) if gaps else None, "tpot_p25": q(0.25),
            "tpot_p75": q(0.75), "tpot_p95": q(0.95), "tpot_mean": (times[-1] - times[0]) / len(gaps) if gaps else None,
            "prefill_est": (times[0] - statistics.median(gaps)) if gaps else None, "total": time.time() - t0, "usage": usage, "finish": finish, "error": error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["state", "preview", "run"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--prompts", default="short:8,medium:32")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if a.cmd == "state":
        st = state()
        print(json.dumps({"nodes": list(st["topology"]["nodes"]), "node_memory": st.get("nodeMemory"), "node_backends": st.get("nodeBackends"), "instances": list(st["instances"])}, indent=1)[:4000])
        return
    if a.cmd == "preview":
        r = call("GET", "/instance/previews?model_id=" + a.args[0])
        for p in r["previews"]:
            inst = p.get("instance")
            lay = None
            if inst and "PipelineShardMetadata" in json.dumps(inst):
                kind, sh = layout_of(inst)
                lay = [(s["device_rank"], s["start_layer"], s["end_layer"], s["node"][:8]) for s in sh]
            print(p["sharding"], p["instance_meta"], "layout", lay, "delta", p.get("memory_delta_by_node"), "error", p.get("error"))
        if a.out:
            json.dump(r, open(a.out, "w"), indent=1)
        return
    label, meta, model = a.args
    st0 = state()
    out = {"label": label, "meta": meta, "model": model, "node_memory": st0.get("nodeMemory"), "runs": {}}
    iid, inst, t_place = place(meta, model)
    kind, sh = layout_of(inst)
    out.update({"kind": kind, "layout": sh, "place_s": t_place, "instance": inst})
    print(f"placed {kind}: " + ", ".join(f"rank{s['device_rank']}={s['node'][:8]} layers {s['start_layer']}-{s['end_layer']} ({s['n_layers']})" for s in sh), flush=True)
    try:
        t_load, tags = wait_ready(iid)
        out["load_s"] = t_load
        print(f"ready after {t_load:.1f}s", flush=True)
        P = prompts()
        for spec in a.prompts.split(","):
            name, n = spec.split(":")
            res = []
            for _ in range(a.reps):
                r = chat_stream(model, P[name], int(n))
                res.append(r)
                print(f"  {name} n={n}: ttft {r['ttft']}, tpot median {r['tpot_median']}, p95 {r['tpot_p95']}, usage {r['usage']}, finish {r['finish']}, error {r['error']}, text {r['text'][:80]!r}", flush=True)
            out["runs"][f"{name}:{n}"] = res
    finally:
        if not a.keep:
            try:
                out["delete_s"] = delete(iid)
            except Exception as e:  # noqa: BLE001
                out["delete_error"] = str(e)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
