"""The expected cross-host decode step from two single-machine profiles, WITHOUT the link.

    python benchmarks/compose_expected.py --stage0 linux_profile.json --stage1 mac_profile.json [--measured-ms X]

Profiles come from `distributed_timeline.py --profile` on loopback runs. Rank 0's side (resume, sampler, graph build, compute, send prep) comes from the
stage-0 machine's profile, rank 1's side (sampler, receive posting, first use, compute, pre-gather) from the stage-1 machine's, and the software transit and
AllGather transfer from a loopback run (default: the stage-1 machine's, whose receive and gather bridge they include). Then
    expected = max(rank0 time to send, rank1 time to receive-posted) + transit + first-use1 + compute1 + pre-gather1 + ag-xfer
Anything the physical run adds beyond this is the link and the cross-host effects the physical run is meant to localize.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--hop-from", default="", help="profile whose transit/ag_xfer to use (default: --stage1)")
    ap.add_argument("--measured-ms", type=float, default=0.0)
    a = ap.parse_args()
    p0, p1 = json.load(open(a.stage0)), json.load(open(a.stage1))
    hop = json.load(open(a.hop_from)) if a.hop_from else p1
    path0 = p0["resume0"] + p0["sampler0"] + p0["pre_compute0"] + p0["compute0"] + p0["send_prep0"]
    path1 = p1["resume1"] + p1["sampler1"] + p1["pre_recv1"]
    expected = max(path0, path1) + hop["transit"] + p1["first_use1"] + p1["compute1"] + p1["pre_gather1"] + hop["ag_xfer"]
    print(f"rank 0 reaches its send after {path0:.0f} us (stage-0 profile); rank 1 has its receive posted after {path1:.0f} us (stage-1 profile) -> critical side: {'rank 0' if path0 >= path1 else 'rank 1'}")
    print(f"+ software transit {hop['transit']:.0f} + first-use {p1['first_use1']:.0f} + stage-1 compute {p1['compute1']:.0f} + pre-gather {p1['pre_gather1']:.0f} + AllGather transfer {hop['ag_xfer']:.0f}")
    print(f"expected step without the link: {expected:.0f} us" + (f"; measured {a.measured_ms*1000:.0f} us -> beyond local components: {a.measured_ms*1000-expected:.0f} us" if a.measured_ms else ""))


if __name__ == "__main__":
    main()
