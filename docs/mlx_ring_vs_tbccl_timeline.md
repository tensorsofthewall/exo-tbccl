# One decode token: MlxRing vs MlxTbccl, Linux CUDA and macOS Metal

Source: `benchmarks/sync_recorder.py` + `sync_timeline.py` on the Phase 54 driver (`real_model_two_host.py`, Qwen3-0.6B-8bit, split 21/7, medium prompt, 48 decode tokens, loopback, two processes). Data: `docs/data/phase57/`. All numbers are the **representative (median-length) steady-state token**, top-level events only, microseconds, offsets from the first `step_complete`. Vocabulary (one for all four views):

* `compute` = an `mx.eval` whose time is GPU stage work (`model_output_eval`) or the sampler/next-token graph (`token_eval`, the driver's `mx.eval(t)`);
* `wait` = host blocked for data that another rank produces (it contains that rank's compute): `recv wait`, `gather wait`;
* `bridge` = TbcclPipelineComm-only host work (borrow, view eval, DLPack export, destination allocation, call overhead);
* `sync` = an `mx.eval` that waits for nothing (status check).

Loopback shares one machine between the ranks, so a rank's `wait` is mostly the other rank's compute: the **critical path of a token is the sum of the compute and the non-overlapped host work on both ranks**.

## Structural difference (same for both platforms)

```
MlxRing                                                    TbcclPipelineComm
rank 1:  eval(x) [template]                                eval(x) [template]
         recv_like -> lazy Recv node (1 us)                recv_like: zeros+eval(dest) [bridge] ; borrow ; native recv ; WAIT for rank 0
         eval(x)   --- runs Recv on CPU-stream thread,     eval(x) -> no-op (status check)
                   --- WAITS for rank 0's data
         stage compute: eval(output)                       stage compute: eval(output)
         all_gather -> lazy node                           all_gather: zeros+eval(dest) ; 2 borrows ; native collective ; WAIT
         eval(output) --- runs AllGather, WAITS            eval(output) -> no-op
rank 0:  stage compute: eval(output)                       stage compute: eval(output)
         send -> lazy Send node                            send: borrow ; native send ; WAIT (until bytes are in the socket)
         eval(output) --- runs Send on CPU-stream thread   eval(output) -> no-op
         all_gather -> lazy; eval(output) --- WAITS        all_gather: ... WAIT
```

Neither side overlaps communication with GPU work: MlxRing postpones the transfer to the next `mx.eval`, which exo calls immediately; TbcclPipelineComm performs and waits for it inside the call. The same waits happen; what differs is the **bridge** work TbcclPipelineComm adds around them and the thread that performs the transfer (the ring's CPU-stream worker vs TBCCL's lane threads).

## Linux CUDA, loopback

```
MlxRing   rank 0  (token 7399 us)
  +0      step_complete
  +13     send (lazy node)                                   2 us
  +23     eval send_dependency  [ring SEND runs here]       920 us   <- waits for the socket write / reader pace
  +948    eval cache_dependency                               9 us
  +959    all_gather (lazy)                                   2 us
  +968    eval post_allgather   [ring ALLGATHER runs here] 3007 us   <- waits for rank 1's compute + gather
  +4032   eval token (sampler)                              587 us
  +5209   eval model_output  [stage 0 compute]             2182 us
MlxTbccl  rank 0  (token 6308 us)
  +0      step_complete
  +5      send: bridge+native, WAIT                         188 us   (borrow ~52 us, view eval ~13, export ~13, native wait 108)
  +205    eval send_dependency (sync)                         2 us
  +212    eval cache_dependency                              17 us
  +232    all_gather: bridge+native, WAIT                  2275 us   (zeros eval 159 us, 3 borrows ~133 us, native wait 1990 us = peer compute + gather)
  +2523   eval post_allgather (sync)                         19 us
  +2589   eval token (sampler)                             1033 us
  +4213   eval model_output [stage 0 compute]              2088 us
MlxRing   rank 1  (token 7438 us)
  +26     eval post_allgather [ring ALLGATHER]              512 us
  +609    eval token                                        614 us
  +1258   eval pre_recv_template                            707 us
  +1972   eval post_recv  [ring RECV runs here]            3551 us   <- waits for rank 0's compute+send
  +6038   eval model_output [stage 1 compute]              1389 us
MlxTbccl  rank 1  (token 6318 us)
  +7      all_gather: bridge+native, WAIT                   407 us
  +435    eval post_allgather (sync)                         28 us
  +535    eval token                                        627 us
  +1200   eval pre_recv_template                            508 us
  +1717   recv_like: bridge+native, WAIT                   3026 us   (zeros eval 44 us, borrow, native wait 2822 us)
  +4752   eval post_recv (sync)                               3 us
  +5267   eval model_output [stage 1 compute]             1039 us
```

On Linux loopback the ring is **slower** (7.4 ms vs 6.3 ms): its `send_dependency_eval` (920-1100 us) and gather waits are longer than TBCCL's native waits, and the stage-1 `model_output_eval` is 1.4 ms vs 1.0 ms. The bridge costs TBCCL about 0.3-0.4 ms per token here (borrows 133-168 us, a fresh-destination eval 44-159 us, export 17-38 us) and the ring's CPU-stream transfer path costs more than that.

## macOS Metal, loopback

```
MlxRing   rank 0  (token 10902 us)
  +2      send (lazy)                                        1 us
  +6      eval send_dependency  [ring SEND]                 25 us
  +34     eval cache_dependency                             22 us
  +58     all_gather (lazy)                                  1 us
  +63     eval post_allgather   [ring ALLGATHER]          2227 us   <- peer's compute + gather
  +2308   eval token                                       3382 us
  +6032   eval model_output [stage 0 compute]             4868 us
MlxTbccl  rank 0  (token 11398 us)
  +1      send: bridge+native, WAIT                          42 us   (borrow ~28, export 1.5, wait 10)
  +47     eval send_dependency (sync)                         0.4 us
  +50     eval cache_dependency                              23 us
  +76     all_gather: bridge+native, WAIT                  2315 us   (zeros eval 166 us, 3 borrows ~100 us, native wait 2086 us)
  +2398   eval post_allgather (sync)                         22 us
  +2443   eval token                                       3233 us
  +6089   eval model_output [stage 0 compute]             5307 us
MlxRing   rank 1  (token 10842 us)
  +7      eval post_allgather   [ring ALLGATHER]            34 us
  +59     eval token                                       3411 us
  +3477   eval pre_recv_template                             189 us
  +3670   eval post_recv  [ring RECV]                      5029 us   <- waits for rank 0's compute+send
  +8826   eval model_output [stage 1 compute]             2014 us
MlxTbccl  rank 1  (token 11401 us)
  +1      all_gather: bridge+native, WAIT                   259 us   (zeros eval 171 us, borrows 104 us, wait 22 us)
  +267    eval post_allgather (sync)                         18 us
  +309    eval token                                       3616 us
  +3936   eval pre_recv_template                            181 us
  +4118   recv_like: bridge+native, WAIT                   5259 us   (zeros eval 161 us, native wait 5083 us)
  +9381   eval post_recv (sync)                               0.4 us
  +9513   eval model_output [stage 1 compute]             1886 us
```

On Mac loopback the ring pays almost nothing for its transfers (send_dependency 25 us, rank 1 all_gather 34 us), while TBCCL pays the bridge visibly: **three fresh-destination allocations of ~160-170 us each** (rank 0 gather destination, rank 1 receive and gather destinations) plus ~100 us of borrows on each rank: about 0.6 ms per token on the critical path (rank 0's send+gather bridge ~0.31 ms, rank 1's recv+gather ~0.45 ms, partly overlapping). That matches the measured step difference exactly: **TBCCL 11.51 ms vs ring 10.97 ms median TPOT (+0.55-0.62 ms)**. The null-transport control (shared memory, the same eval/materialize pattern, no bridge) is within +0.03 ms of the ring.

## Where the GPU, CPU and network are active

GPU: only inside `model_output_eval` (stage compute), the sampler's `token` eval and the small allocation/view kernels; during every `wait` the GPU is idle (the stage that is waiting has nothing queued, the stage that is computing is on the other rank). CPU: busy in `bridge` and in the spun CUDA/Metal waits; the network is active only inside `recv`/`send`/`gather` waits and is a few tens of microseconds of each. **No variant overlaps communication with GPU work**; they differ in who sleeps and which thread moves the bytes.
