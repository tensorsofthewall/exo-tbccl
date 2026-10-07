"""Per-token critical-path components of a two-stage decode from sync_recorder dumps, and a no-link composition for a heterogeneous pair.

    python benchmarks/critical_path.py --stage0 rank0.json --stage1 rank1.json [--label NAME]
    python benchmarks/critical_path.py --compose --stage0 linux_rank0.json --stage1 mac_rank1.json [--label NAME]

Steady-state decode tokens (first 3 dropped), medians, microseconds. Per rank the dump's top-level events are bucketed as:
  model     model_output_eval (the stage's GPU computation)
  sampler   the driver's token eval (every rank samples; they run concurrently after the all_gather)
  pre_recv  pre_recv_template_eval (evaluates the stage input before the receive; rank 1)
  cache     cache_dependency_eval
  bridge    TbcclPipelineComm only: each communication call minus its native wait (borrow, view eval, DLPack export, destination allocation, call overhead)
  comm      transfer + wait: tbccl native waits, or for MlxRing the evals that execute the lazy Send/Recv/AllGather (send_dependency, post_recv, post_allgather)
  sync      TbcclPipelineComm's post-communication evals (status checks)
The no-link critical path of one token (both ranks synchronize through the receive and the all_gather) is
    T = max(sampler0 + model0 + cache0, sampler1 + pre_recv1) + bridge0(send) + hop1 + bridge1(recv) + model1 + bridge1(gather) + hop2 + sync
and a measured same-machine run validates the pieces (the T printed next to the measured TPOT). `--compose` builds T from stage 0 of one dump pair and stage 1 of
another, i.e. a Linux stage 0 with a Mac stage 1 (or the reverse) WITHOUT the link: the physical value minus this is what the local evidence cannot explain.
hop1/hop2 are the loopback transfer latencies (ring: execution inside the dependent evals; tbccl: native waits not dominated by the peer's compute).
"""
import argparse
import json
import statistics


def load(path):
    d = json.load(open(path))
    ev = [dict(t0=a, t1=b, kind=k, label=l, depth=dp) for a, b, k, l, dp, *_ in d["events"]]
    dec = [e for e in ev if not e["label"].startswith("prefill:")]
    marks = [i for i, e in enumerate(dec) if e["kind"] == "comm" and e["label"] == "step_complete"]
    spans = [dec[a: b + 1] for a, b in zip(marks, marks[1:])][3:]
    return d["rank"], d["backend"], spans


def dur(e):
    return (e["t1"] - e["t0"]) / 1000.0


def comp(path):
    rank, backend, spans = load(path)
    rows = []
    for sp in spans:
        c = dict(model=0, sampler=0, pre_recv=0, cache=0, bridge=0, comm=0, sync=0, wait_all_gather=0, wait_recv=0, wait_send=0, token_total=(sp[-1]["t1"] - sp[0]["t0"]) / 1000.0)
        for e in sp:
            lab, kind = e["label"], e["kind"]
            if kind == "eval" and lab == "model_output_eval": c["model"] += dur(e)
            elif kind == "eval" and lab == "real_model_loopback.py:worker#4": c["sampler"] += dur(e)
            elif kind == "eval" and lab == "pre_recv_template_eval": c["pre_recv"] += dur(e)
            elif kind == "eval" and lab == "cache_dependency_eval": c["cache"] += dur(e)
            elif kind == "eval" and lab in ("send_dependency_eval", "post_recv_eval", "post_allgather_eval") and backend == "ring": c["comm"] += dur(e)
            elif kind == "eval" and lab in ("post_recv_eval", "post_allgather_eval", "send_dependency_eval"): c["sync"] += dur(e)
            elif kind == "tbccl_wait":
                c["comm"] += dur(e)
                c["wait_" + lab.split(":")[1]] += dur(e)
            elif kind == "comm" and e["depth"] == 0 and lab in ("send", "recv_like", "all_gather") and backend == "tbccl":
                inner = sum(dur(x) for x in sp if x["kind"] == "tbccl_wait" and x["t0"] >= e["t0"] and x["t1"] <= e["t1"])
                c["bridge"] += dur(e) - inner
        rows.append(c)
    return rank, backend, {k: statistics.median(r[k] for r in rows) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    r0, b0, c0 = comp(a.stage0)
    r1, b1, c1 = comp(a.stage1)
    print(f"{a.label}: stage0={a.stage0.rsplit('/',1)[-1]} ({b0})  stage1={a.stage1.rsplit('/',1)[-1]} ({b1})")
    print(f"  {'component':10}{'stage0 us':>11}{'stage1 us':>11}")
    for k in ("model", "sampler", "pre_recv", "cache", "bridge", "comm", "sync"):
        print(f"  {k:10}{c0[k]:11.0f}{c1[k]:11.0f}")
    pre = max(c0["sampler"] + c0["model"] + c0["cache"], c1["sampler"] + c1["pre_recv"])
    # no-link period: the pre-send work of the slower stage, then stage 1's model, plus both ranks' bridge (it sits on the path twice: send/recv hop, gather hop)
    # and the transfer waits that are not peer compute (small: measured separately by the hop waits)
    path = pre + c1["model"] + c0["bridge"] + c1["bridge"] + c0["sync"] + c1["sync"]
    print(f"  no-link critical path: max(stage0 pre-send {c0['sampler']+c0['model']+c0['cache']:.0f}, stage1 pre-recv {c1['sampler']+c1['pre_recv']:.0f}) + stage1 model {c1['model']:.0f}"
          f" + bridge {c0['bridge']+c1['bridge']:.0f} + sync {c0['sync']+c1['sync']:.0f} = {path:.0f} us")
    print(f"  recorded token length (stage0 / stage1 rank): {c0['token_total']:.0f} / {c1['token_total']:.0f} us;"
          f" residual (transfer latency + wakeups not in the buckets) = {(c0['token_total'] + c1['token_total']) / 2 - path:.0f} us")


if __name__ == "__main__":
    main()
