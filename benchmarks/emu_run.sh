#!/bin/sh
# One real-Mac-stage + emulated-Linux-peer run (Mac loopback).
#   usage: emu_run.sh <A|B> <tbccl|ring> <port> <outprefix> [tokens] [extra real-worker args...]
# Run from the exo-tbccl checkout with the exo venv's python in $PY (default ../exo/.venv/bin/python). The real Mac stage records with EXO_P57_SYNC=<outprefix>;
# the emulator writes <outprefix>.rank<R>.json and <outprefix>.fidelity.json. Extra env (EXO_TBCCL_ALLOC_STREAM=cpu, EXO_P59_*) is inherited by both processes.
O=$1; B=$2; PORT=$3; OUT=$4; TOK=${5:-48}; shift 5 2>/dev/null
PY=${PY:-../exo/.venv/bin/python}; PROFILES=${PROFILES:-benchmarks/emulator_profiles}
export HF_HUB_OFFLINE=1 EXO_OFFLINE=true
if [ "$O" = A ]; then REAL=1; SPLIT=21; else REAL=0; SPLIT=7; fi
COMMON="--host 127.0.0.1 --peer 127.0.0.1 --port $PORT"
if [ -n "$EXO_P59_EXT_SPIN" ]; then  # Phase 59 control: a CPU-burning process OUTSIDE the pipeline process (a different thread group)
  python3 -c "import time,sys\nt=time.time()\nwhile time.time()-t<600: pass" &
  SPID=$!
fi
$PY benchmarks/remote_peer_emulator.py --orientation $O --backend $B --profile $PROFILES/profile_${O}_$B.json $COMMON --tokens $TOK ${EMU_ARGS} --out $OUT > $OUT.emu.out 2>&1 &
EP=$!
if [ -n "$SYNTH_MS" ]; then  # Phase 59 control: a synthetic Metal stage (orientation B only) instead of Qwen
  $PY benchmarks/synthetic_mac_stage.py --backend $B --stage-ms $SYNTH_MS --sampler-ms $SYNTH_MS --tokens $TOK $COMMON --out $OUT > $OUT.real.out 2>&1
else
  EXO_P57_SYNC=$OUT $PY benchmarks/real_model_two_host.py --rank $REAL $COMMON --split $SPLIT --prompt medium --tokens $TOK --chunk 512 --backend $B --ring-ips 127.0.0.1,127.0.0.1 "$@" > $OUT.real.out 2>&1
fi
wait $EP
[ -n "$SPID" ] && kill $SPID 2>/dev/null
grep -o '"tpot_ms": [0-9.]*' $OUT.real.out | head -1 | sed "s/.*: //" | sed "s/^/\"tpot_ms\": /"
