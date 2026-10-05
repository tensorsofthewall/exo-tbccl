"""Component-by-component comparison of the physical cross-host timeline traces with the emulator runs (same decomposition, same tool).

    python benchmarks/emulator_vs_physical.py --phys r0 r1 [r0 r1 ...] --emu r0 r1 [...] [--label "A tbccl"]

Medians over all steps of all runs of each set (microseconds) for the components the emulator is supposed to reproduce: the peer being ready, transit, the
AllGather entry skew and the AllGather completion, plus every Mac-executed row, so the table shows what the emulator reproduces and what it does not.
"""
import argparse
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import compare_runs as cr  # noqa: E402

KEYS = [("peer-late (peer-ready delay)", "peer-late"), ("transit (comm)", "transit"), ("AG entry skew", "ag-entry-skew"), ("AG completion (xfer)", "ag-xfer"),
        ("first-use", "first-use"), ("rank 0 sampler", "sampler"), ("rank 0 resume", "resume"), ("rank 0 graph build", "pre-compute"), ("rank 0 compute", "compute0"),
        ("rank 1 compute", "compute1"), ("step period", "period")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phys", nargs="+", required=True)
    ap.add_argument("--emu", nargs="+", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    P = [cr.steps(a.phys[i], a.phys[i + 1])[0] for i in range(0, len(a.phys), 2)]
    E = [cr.steps(a.emu[i], a.emu[i + 1])[0] for i in range(0, len(a.emu), 2)]
    print(f"{a.label}: physical {len(P)} run(s), emulator {len(E)} run(s)")
    print(f"  {'component':32}{'physical':>10}{'emulator':>10}{'difference':>12}")
    for name, k in KEYS:
        pv = statistics.median(c[k] for run in P for c in run.values())
        ev = statistics.median(c[k] for run in E for c in run.values())
        print(f"  {name:32}{pv:10.0f}{ev:10.0f}{ev - pv:+12.0f}")


if __name__ == "__main__":
    main()
