# Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| placement error when selecting `MlxTbccl` | `exo_tbccl` does not import | `python -c "import exo_tbccl; print(exo_tbccl.is_available())"` prints the reason |
| bootstrap times out (120 s) | a rank did not publish its endpoint or its address is unreachable | check the address exo chose per node and that every runner started |
| `TbcclProtocolMismatchError` or `protocol_mismatch ... wire protocol` | the two hosts' TBCCL installs have different wire protocol versions | rebuild exo-tbccl on both against the same TBCCL prefix |
| the exo process does not exit after an `MlxTbccl` instance | the runner byte-exchange inbox was not drained (fixed in the exo integration; older exo branches lack the fix) | use the exo version recorded in [compatibility](../reference/compatibility.md) |
| a send error appears later than the send | `EXO_TBCCL_ASYNC_SEND=1` holds a failed send until the next communication point | expected; the error is raised no later than the next synchronizing call |

A peer that dies surfaces as a structured transport error on the survivors within about a second; the runner fails and exo reports an error to the client. Cancellation never fails the instance.
