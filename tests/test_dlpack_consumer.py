import ctypes

import pytest

from exo_tbccl._loader import native
from tests.fake_dlpack import FakeProducer


def buf(n=64):
    return (ctypes.c_uint8 * n)(*range(n))


def test_contiguous_export_reports_ptr_nbytes_and_runs_deleter_once():
    b = buf()
    p = FakeProducer(b, [4, 8], bits=16)  # 4*8*2 bytes
    e = native.Export(p)
    assert e.ptr == ctypes.addressof(b) and e.nbytes == 64 and e.device == (1, 0)
    assert p.deleted == 0 and not e.released
    e.release()
    e.release()
    assert p.deleted == 1 and e.released


def test_deleter_runs_when_the_export_is_garbage_collected():
    p = FakeProducer(buf(), [8])
    e = native.Export(p)
    del e
    assert p.deleted == 1


def test_byte_offset_and_c_strides_are_honoured():
    b = buf()
    p = FakeProducer(b, [2, 3], strides=[3, 1], bits=8, byte_offset=5)
    e = native.Export(p)
    assert e.ptr == ctypes.addressof(b) + 5 and e.nbytes == 6
    e.release()


def test_non_contiguous_is_rejected_and_the_deleter_still_runs():
    p = FakeProducer(buf(), [4, 4], strides=[1, 4])
    with pytest.raises(ValueError, match="C-contiguous"):
        native.Export(p)
    assert p.deleted == 1


def test_size_one_dims_ignore_their_stride():
    p = FakeProducer(buf(), [1, 4, 1], strides=[999, 1, 777])
    e = native.Export(p)
    assert e.nbytes == 4
    e.release()


def test_overflowing_byte_count_is_rejected():
    p = FakeProducer(buf(), [2**62, 8], bits=32)
    with pytest.raises(OverflowError):
        native.Export(p)
    assert p.deleted == 1


def test_sub_byte_dtypes_round_up():
    p = FakeProducer(buf(), [7], bits=4)
    e = native.Export(p)
    assert e.nbytes == 4
    e.release()


def test_null_data_with_payload_is_rejected_but_empty_is_fine():
    with pytest.raises(ValueError, match="null"):
        native.Export(FakeProducer(buf(), [8], null_data=True))
    e = native.Export(FakeProducer(buf(), [0], null_data=True))
    assert e.nbytes == 0
    e.release()


def test_invalid_dtype_is_rejected():
    with pytest.raises(ValueError, match="dtype"):
        native.Export(FakeProducer(buf(), [8], bits=0))


def test_method_device_overrides_a_cpu_capsule_like_mlx():
    p = FakeProducer(buf(), [8], device=(1, 0), method_device=(8, 0))
    e = native.Export(p)
    assert e.device == (8, 0)
    e.release()


def test_conflicting_gpu_devices_are_rejected():
    p = FakeProducer(buf(), [8], device=(2, 0), method_device=(8, 0))
    with pytest.raises(ValueError, match="mismatch"):
        native.Export(p)
    assert p.deleted == 1


def test_non_capsule_producer_is_a_type_error():
    class Bad:
        def __dlpack__(self):
            return 42

    with pytest.raises(TypeError):
        native.Export(Bad())
