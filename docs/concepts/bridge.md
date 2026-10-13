# The MLX bridge (DLPack)

Arrays enter TBCCL through the Python DLPack protocol (`__dlpack_device__`, `__dlpack__`) and the DLPack C structures, never through MLX internals.

Findings that shape the bridge (MLX 0.32; item 5 differs between MLX builds):

1. `mx.array.__dlpack_device__()` reports the real device: a Linux CUDA build reports `kDLCUDAManaged` (13), because MLX CUDA arrays are CUDA managed memory, and a Mac reports `kDLMetal` (8). The bridge therefore maps 1 and 3 to host memory, 2 and 13 to CUDA, and 8 to Metal-shared memory.
2. `__dlpack__()` goes through NumPy's buffer path: the capsule's device is always `kDLCPU` and `bfloat16` cannot be exported. The device must come from `__dlpack_device__`, not the capsule.
3. `arr.view(<same-width unsigned dtype>)` is a zero-copy alias for float32, float16 and bfloat16. Exporting that view makes the transport dtype-agnostic (quantized and fp8 payloads included) and lets row-contiguity be detected from the exported strides.
4. The adapter evaluates the array, makes it row-contiguous when its strides are not C-order, takes the unsigned view, exports it, keys the memory kind off the device, and keeps the array, the view and the export alive until the TBCCL `Work` is terminal.

5. For a Metal array the exported `data` is either the CPU-visible contents address (MLX 0.32.0 development builds) or the `id<MTLBuffer>` handle with the array's offset in `byte_offset` (the 0.32.4 development build and later; DLPack's own convention). The two are indistinguishable by type, and reading a handle as an address corrupts memory. The bridge probes the convention once per process by exporting a known byte pattern, resolves a handle to its contents address through the Objective-C runtime (`[buffer contents] + byte_offset`, shared storage only), and raises `TbcclUnsupportedError` when the export is neither; nothing is guessed. The receive pool's poison control goes through the same resolution.

CPU-visible bytes are coherent after `mx.eval` on both platforms, and TBCCL's Metal-shared and CUDA providers accept these pointers end to end. A producer's DLPack deleter may run Python code, so it is never called with an exception pending. Do not `fork` a process that already imported MLX or CUDA; use `spawn`.
