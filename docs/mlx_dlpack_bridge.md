# MLX to TBCCL array bridge: spike findings (MLX 0.32.0, Linux CUDA 13 and macOS Metal)

Spike scripts: `docs/spikes/dl_spike.py`, `dl_spike2.py` (ctypes view of the DLPack structs; run on both hosts).

1. `mx.array.__dlpack_device__()` reports the real device: Linux CUDA build `(13, 0)` = `kDLCUDAManaged`
   (MLX CUDA arrays are CUDA managed memory, NOT `kDLCUDA`), Mac `(8, 0)` = `kDLMetal`. The plan's device table must
   therefore also accept `kDLCUDAManaged` -> `TBCCL_MEMORY_CUDA`.
2. `mx.array.__dlpack__()` goes through NumPy's buffer path: the capsule's `DLTensor.device` is always `kDLCPU`
   (even for GPU-backed arrays), `bfloat16` raises `TypeError: bfloat16 arrays cannot be converted to NumPy`, and
   the `data` pointer is the array's real host-addressable storage (slices keep the same buffer, `data` moves by
   the slice offset; transposes export strides `[1, 4]`, no copy). So the device must come from
   `__dlpack_device__`, not from the capsule.
3. `arr.view(mx.uint8)` is a zero-copy alias on both platforms for float32/float16/bfloat16: its capsule `data`
   equals the storage pointer, bytes match the NumPy reference, and writing through the pointer changes the
   original array (verified with a bfloat16 destination). This gives a dtype-independent byte view, which fixes the
   bfloat16 gap and makes the transport dtype-agnostic (quantized/fp8 payloads included).
4. Consequences for the adapter: evaluate the array, make it row-contiguous (`mx.contiguous`) when its exported
   strides are not C-order, take `view(uint8)`, evaluate it, export DLPack, key the TBCCL memory kind off
   `__dlpack_device__` (1 -> HOST, 13 -> CUDA, 8 -> METAL_SHARED), keep the array, the view and the capsule alive
   until the TBCCL Work is terminal.
5. Not yet verified (next steps): TBCCL `METAL_SHARED` acceptance of this pointer end to end, CUDA-managed
   pointer as `TBCCL_MEMORY_CUDA` end to end, whether the CPU-visible bytes are coherent with prior GPU writes
   after `mx.eval` on both platforms (the byte-equality check above ran after `mx.eval` and passed).
