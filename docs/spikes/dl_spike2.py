import ctypes, mlx.core as mx
exec(open(__file__.replace("dl_spike2","dl_spike")).read().split("a=mx.arange(16")[0])
def ptr(x):
    t,cap=info(x); return t.data+t.byte_offset, t.device.device_type, cap, t
for dt in (mx.float32, mx.float16, mx.bfloat16):
    x=(mx.arange(24)*1).astype(dt).reshape(2,3,4); mx.eval(x)
    v=x.view(mx.uint8); mx.eval(v)
    p,d,cap,t=ptr(v)
    print(dt, "dev", x.__dlpack_device__(), "view shape", v.shape, "nbytes", x.nbytes, "view nbytes", v.nbytes)
    raw=(ctypes.c_uint8*x.nbytes).from_address(p); 
    import numpy as np
    got=bytes(raw)
    ref=bytes(np.array(x.view(mx.uint16) if dt==mx.bfloat16 else x).tobytes()) if dt!=mx.bfloat16 else bytes(np.array(x.view(mx.uint16)).tobytes())
    print(" bytes equal numpy ref:", got==ref)
# recv-style write test
dst=mx.zeros((4,4),dtype=mx.bfloat16); mx.eval(dst)
v=dst.view(mx.uint8); mx.eval(v); p,d,cap,t=ptr(v)
ctypes.memmove(p, bytes(range(32)), 32)
print(" dst after memmove:", bytes(__import__('numpy').array(dst.view(mx.uint8)))[:8], dst.dtype)
# odd tail: uint8 length 7
o=mx.arange(7,dtype=mx.uint8); mx.eval(o); print(" odd", ptr(o)[3].shape[0] if False else o.nbytes)
