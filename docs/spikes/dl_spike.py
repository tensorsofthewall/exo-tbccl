import ctypes, sys
import mlx.core as mx
class DLDevice(ctypes.Structure): _fields_=[("device_type",ctypes.c_int32),("device_id",ctypes.c_int32)]
class DLDataType(ctypes.Structure): _fields_=[("code",ctypes.c_uint8),("bits",ctypes.c_uint8),("lanes",ctypes.c_uint16)]
class DLTensor(ctypes.Structure): _fields_=[("data",ctypes.c_void_p),("device",DLDevice),("ndim",ctypes.c_int32),("dtype",DLDataType),("shape",ctypes.POINTER(ctypes.c_int64)),("strides",ctypes.POINTER(ctypes.c_int64)),("byte_offset",ctypes.c_uint64)]
class DLManaged(ctypes.Structure): _fields_=[("dl_tensor",DLTensor),("manager_ctx",ctypes.c_void_p),("deleter",ctypes.c_void_p)]
ctypes.pythonapi.PyCapsule_GetPointer.restype=ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes=[ctypes.py_object,ctypes.c_char_p]
def info(a):
    mx.eval(a)
    cap=a.__dlpack__()
    p=ctypes.pythonapi.PyCapsule_GetPointer(cap,b"dltensor")
    m=DLManaged.from_address(p); t=m.dl_tensor
    sh=[t.shape[i] for i in range(t.ndim)]
    st=[t.strides[i] for i in range(t.ndim)] if t.strides else None
    print("dev",(t.device.device_type,t.device.device_id),"data",hex(t.data or 0),"off",t.byte_offset,"dtype",(t.dtype.code,t.dtype.bits,t.dtype.lanes),"shape",sh,"strides",st)
    return t, cap
a=mx.arange(16,dtype=mx.float32)*3; t,cap_a=info(a)
if t.device.device_type in (1,13):
    buf=(ctypes.c_float*16).from_address((t.data or 0)+t.byte_offset); print("host-readable:",list(buf)[:6])
import contextlib
for nm,x in (("bf16",mx.arange(16,dtype=mx.bfloat16)),("f16",mx.arange(16,dtype=mx.float16))):
    try: r=info(x)
    except Exception as ex: print(nm,'ERR',repr(ex))
for nm,x in (("T",a.reshape(4,4).T),("slice",a[4:])):
    try: r=info(x)
    except Exception as ex: print(nm,'ERR',repr(ex))
