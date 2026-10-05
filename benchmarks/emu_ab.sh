#!/bin/sh
# Interleaved paired repetitions of emu_run.sh.
#   usage: emu_ab.sh <outdir> <A|B> <reps> <spec...>     spec = name:backend[:ENV=V,ENV=V]
# Writes <outdir>/<orientation>_<name>_<rep>.{rank0,rank1}.json (+ .fidelity.json) and appends "<name> rep=<n> tpot_ms=<x>" to <outdir>/<orientation>_tpot.txt.
OUTDIR=$1; O=$2; REPS=$3; shift 3
mkdir -p $OUTDIR
n=0; set -- "$@"; cnt=$#
for rep in $(seq 0 $((REPS-1))); do
  for k in $(seq 0 $((cnt-1))); do
    idx=$(( (k + rep) % cnt )); [ $((rep%2)) -eq 1 ] && idx=$(( cnt-1 - idx ))
    i=0; for s in "$@"; do [ $i -eq $idx ] && spec=$s; i=$((i+1)); done
    name=${spec%%:*}; rest=${spec#*:}; b=${rest%%:*}; envs=""; [ "$rest" != "$b" ] && envs=$(echo ${rest#*:} | tr ',' ' ')
    PORT=$((39400 + (n%200)*10)); n=$((n+1))
    t=$(env $envs sh benchmarks/emu_run.sh $O $b $PORT $OUTDIR/${O}_${name}_$rep 48 2>/dev/null | tail -1 | grep -o '[0-9.]*$')
    echo "$name rep=$rep tpot_ms=$t" >> $OUTDIR/${O}_tpot.txt
  done
done
echo DONE >> $OUTDIR/${O}_tpot.txt
