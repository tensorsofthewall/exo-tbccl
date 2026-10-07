# Emulator profiles

`profile_{A,B}_{ring,tbccl}.json` are timing profiles for `benchmarks/remote_peer_emulator.py`: Linux-local intervals (the remote stage's compute and communication gaps per decode step) recorded from physical two-host traces, for orientation A or B of the pipeline and for the `ring` (MlxRing) or `tbccl` backend. The emulator replays them with calibrated sleeps plus a short spin against a real Mac stage over loopback. They contain only timings (no weights and no host details). `benchmarks/emu_run.sh` uses this directory by default (`PROFILES`).
