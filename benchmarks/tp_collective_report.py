"""Phase 67: tables of the exact-size collective results (docs/data/phase67/physical/M1_*, M2_*, both ranks) -> JSON + text.

    python benchmarks/tp_collective_report.py docs/data/phase67/physical [--out docs/data/phase67/collectives.json]
"""
import glob
import json
import sys


def load(d):
    rows = []
    for f in sorted(glob.glob(f"{d}/M[12]_*.out")):
        for line in open(f):
            if line.startswith("{"):
                r = json.loads(line)
                rows.append(r)
    return rows


def main():
    d = sys.argv[1]
    rows = load(d)
    out = {}
    for r in rows:
        out.setdefault(f"{r['op']}|{r['bytes']}|{r['gap_us']}", {})[f"rank{r['rank']}"] = {k: r[k] for k in ("median_us", "p25_us", "p75_us", "p95_us", "n", "cpu_cores", "gap_mode")}
    print(f"{'op':10}{'bytes':>9}{'gap':>5}  {'rank0 med':>9}{'p25':>8}{'p75':>8}{'p95':>8}   {'rank1 med':>9}{'p95':>8}   {'GB/s (rank0, payload/med)':>26}")
    for k, v in sorted(out.items(), key=lambda kv: (kv[0].split('|')[0], int(kv[0].split('|')[1]), int(kv[0].split('|')[2]))):
        op, b, g = k.split("|")
        r0, r1 = v.get("rank0", {}), v.get("rank1", {})
        bw = int(b) / r0["median_us"] / 1e3 if r0 else 0
        print(f"{op:10}{b:>9}{g:>5}  {r0.get('median_us', 0):9.1f}{r0.get('p25_us', 0):8.1f}{r0.get('p75_us', 0):8.1f}{r0.get('p95_us', 0):8.1f}   {r1.get('median_us', 0):9.1f}{r1.get('p95_us', 0):8.1f}   {bw:26.3f}")
    if "--out" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
