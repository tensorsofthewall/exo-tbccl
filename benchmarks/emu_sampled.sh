#!/bin/sh
# Phase 62: emu_run.sh with the external resource sampler attached to the real Mac stage process.
#   usage: emu_sampled.sh <A|B> <tbccl|ring> <port> <outprefix> [tokens]    (env: SAMPLER=0 to run without the sampler; SAMPLE_MS)
O=$1; B=$2; PORT=$3; OUT=$4; TOK=${5:-48}
PY=${PY:-../exo/.venv/bin/python}
if [ "${SAMPLER:-1}" = 1 ]; then
  $PY benchmarks/mac_resource_sampler.py --match real_model_two_host.py --also $PORT --interval-ms ${SAMPLE_MS:-2} --out $OUT.res.json &
  SP=$!
fi
sh benchmarks/emu_run.sh $O $B $PORT $OUT $TOK
[ -n "$SP" ] && wait $SP
