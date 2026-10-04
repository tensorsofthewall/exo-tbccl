# CUDA managed memory, `kDLCUDAManaged` and the host-direct policy (Phase 54)

## Why `kDLCUDAManaged` is special

MLX's CUDA backend reports `__dlpack_device__() == (13, ordinal)`: kDLCUDAManaged. Phase 53 mapped it to `TBCCL_MEMORY_CUDA`, so every decode activation went
through TBCCL's CUDA provider (stream, staging, event completion): ~330 us per 2 KiB round trip against ~31 us when the same storage is described as host memory.
Managed memory is addressable by both processors, so describing it as host memory is possible; whether it is *safe and cheap* depends on the system.

`__dlpack_device__` stays authoritative for what the storage is. The policy only chooses how TBCCL is told to touch it; it never changes the device the bridge
reports, and the export's device is re-checked against the array's on every borrow.

Two MLX details that matter (observed, Phase 54):
- MLX CUDA arrays live in ordinary device memory until their pointer is first requested (`raw_ptr()`, i.e. the DLPack export). At that point MLX allocates managed
  memory, copies, and frees the original (`move_to_unified_memory`, a `cudaMemcpy` plus `cudaFreeAsync`). So the first export of a *fresh* array pays a copy that a
  reused receive buffer does not.
- After that the pointer is a real managed pointer: `cuPointerGetAttribute` reports `is_managed=1`, `memory_type=2` (the driver types managed allocations as device),
  host pointer equal to device pointer.

## Capability inspection (`exo_tbccl/_cuda_caps.py`, internal)

Queried through the CUDA driver API with ctypes (no CUDA build dependency): `cuPointerGetAttribute` (is-managed, memory type, ordinal, host/device pointer) and
`cuDeviceGetAttribute` (`managedMemory`, `concurrentManagedAccess`, `pageableMemoryAccess`, `pageableMemoryAccessUsesHostPageTables`,
`directManagedMemAccessFromHost`, `integrated`, `unifiedAddressing`, compute capability). If the driver cannot be queried the answer is "unknown" and the CUDA path is kept.

This machine (Linux 6.18.34, driver 610.43.02, RTX 3070 Ti Laptop, sm_86): `managedMemory=1 concurrentManagedAccess=1 pageableMemoryAccess=1
pageableMemoryAccessUsesHostPageTables=0 directManagedMemAccessFromHost=0 integrated=0`. The CPU may touch managed pages while the GPU is active, but pages that the
GPU wrote are not directly CPU-mapped: a CPU read migrates them (page fault). That is correct but not free, and the cost grows with the payload.

## Policy

`EXO_TBCCL_CUDA_MANAGED_MODE` (or `FastPathConfig.managed_mode`):

| mode | behavior |
|---|---|
| `cuda` (default) | Phase 53: managed storage is described as CUDA memory. |
| `host` | forced: managed storage is described as host memory, any size. For measurement and tests only. |
| `auto` | host-direct only when the array is CUDA-managed, its device's driver-reported attributes equal a proven signature (`_PROVEN_MANAGED_HOST_SIGNATURES`), and the payload is at most `EXO_TBCCL_MANAGED_MAX_BYTES` (default 8 KiB). Everything else, including an unqueryable driver, keeps the CUDA path. |

`EXO_TBCCL_MANAGED_DIRS=send|recv|send,recv` restricts which directions are remapped (the policy may be asymmetric). Not inferred from the OS, the GPU generation or
`kDLCUDAManaged` alone.

The default stays `cuda`: see `docs/phase54_results.md` for the evidence and for why `auto` is not (yet) the default.

## Host-direct safety requirements

1. The exported storage is managed (verified with the driver, not assumed).
2. The device reports `concurrentManagedAccess=1` (CPU access while kernels run is legal) and `managedMemory=1`.
3. The producer is evaluated before TBCCL reads (`mx.eval` in the bridge, unchanged). No extra `mx.synchronize()` was needed in the stress tests.
4. A receive destination is consumed by the GPU only after the receive Work is terminal.
5. Payload small enough that CPU access to GPU-resident pages is cheap (see results: the break-even on this machine is between 8 and 16 KiB; 64 KiB sends were 2-3x slower
   than the CUDA path, 1 MiB sends ~5x slower).

## Fallback conditions

No driver / query failure; a device whose attributes do not match a proven signature (for example `concurrentManagedAccess=0`, `integrated=1`,
`directManagedMemAccessFromHost=1`, or `pageableMemoryAccess=0`); a non-managed export (plain CUDA, Metal); a payload above the size gate; `cuda` mode. In all of
them the Phase 53 CUDA path is used.

## Migration implications

A GPU-produced buffer read by the CPU migrates to host; a buffer written by the CPU and consumed by the GPU migrates back on first use. The loopback measurements record
this as the "first GPU consumer" time. For decode-sized activations it is within noise of the CUDA path; for MiB-sized prefill activations it is not, which is why `auto`
is size-gated. Explicit prefetch was evaluated; see results.
