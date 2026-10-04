"""A hand-built DLPack producer, so the native consumer's validation and lifetime rules can be tested without any framework."""

import ctypes


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManaged(ctypes.Structure):
    pass


DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManaged))
DLManaged._fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", DELETER)]

PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


class FakeProducer:
    """Describes a tensor over `buffer`; counts deleter calls in `.deleted`."""

    def __init__(self, buffer, shape, strides=None, *, bits=8, lanes=1, code=1, byte_offset=0, device=(1, 0), method_device=None, null_data=False):
        self.buffer = buffer
        self.shape_arr = (ctypes.c_int64 * len(shape))(*shape)
        self.strides_arr = (ctypes.c_int64 * len(shape))(*strides) if strides is not None else None
        self.params = (bits, lanes, code, byte_offset, device, null_data)
        if method_device is not None:
            self.__dlpack_device__ = lambda: method_device
        self.deleted = 0
        self._keep = []

    def __dlpack__(self, *args, **kwargs):
        bits, lanes, code, off, device, null_data = self.params
        m = DLManaged()
        t = m.dl_tensor
        t.data = None if null_data else ctypes.addressof(self.buffer)
        t.device = DLDevice(*device)
        t.ndim = len(self.shape_arr)
        t.dtype = DLDataType(code, bits, lanes)
        t.shape = ctypes.cast(self.shape_arr, ctypes.POINTER(ctypes.c_int64))
        t.strides = ctypes.cast(self.strides_arr, ctypes.POINTER(ctypes.c_int64)) if self.strides_arr is not None else None
        t.byte_offset = off

        def deleter(_p):
            self.deleted += 1

        cb = DELETER(deleter)
        m.deleter = cb
        self._keep.append((m, cb))
        return PyCapsule_New(ctypes.addressof(m), b"dltensor", None)
