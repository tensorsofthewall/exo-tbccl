/* The DLPack v0.8 C structures (https://dmlc.github.io/dlpack/latest/c_api.html), declared here so the bridge needs no framework headers. */
#ifndef EXO_TBCCL_DLPACK_H
#define EXO_TBCCL_DLPACK_H
#include <stdint.h>

#define EXO_DL_CPU 1
#define EXO_DL_CUDA 2
#define EXO_DL_CUDA_HOST 3
#define EXO_DL_METAL 8
#define EXO_DL_CUDA_MANAGED 13

typedef struct { int32_t device_type; int32_t device_id; } ExoDLDevice;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } ExoDLDataType;
typedef struct {
    void *data;
    ExoDLDevice device;
    int32_t ndim;
    ExoDLDataType dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
} ExoDLTensor;
typedef struct ExoDLManagedTensor {
    ExoDLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct ExoDLManagedTensor *self);
} ExoDLManagedTensor;

#endif
