/* exo_tbccl._native: a thin CPython binding of the TBCCL C ABI v1 (include/tbccl/tbccl.h) plus a DLPack consumer.
 *
 * Design rules:
 *   - Only the stable C ABI is used; no TBCCL C++ header or symbol.
 *   - Every blocking call (bootstrap completion, Work waits) releases the GIL.
 *   - No Python finalizer is needed for correctness: handles have explicit close(); the dealloc paths only release what close() would.
 *   - Errors are raised through a Python factory (set_error_factory) from the structured result code, never by parsing text.
 *   - The DLPack consumer follows the protocol: the capsule is renamed "used_dltensor" and the producer's deleter runs on release().
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <tbccl/tbccl.h>

#include "dlpack.h"

static PyObject *g_error_factory = NULL;

static void raise_result(tbcclResult_t code, const char *op, const char *detail) {
    PyObject *exc = NULL;
    if (g_error_factory != NULL) {
        exc = PyObject_CallFunction(g_error_factory, "isz", (int)code, op, detail);
    }
    if (exc == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "%s: tbccl result %d (%s)", op, (int)code, tbcclGetResultString(code));
        }
        return;
    }
    PyErr_SetObject((PyObject *)Py_TYPE(exc), exc);
    Py_DECREF(exc);
}

#define CHECK_API(call, op)                                                       \
    do {                                                                          \
        tbcclResult_t _r = (call);                                                \
        if (_r != TBCCL_SUCCESS) {                                                \
            raise_result(_r, (op), NULL);                                         \
            goto fail;                                                            \
        }                                                                         \
    } while (0)

/* ------------------------------------------------------------------ Export: a consumed DLPack tensor */

typedef struct {
    PyObject_HEAD
    ExoDLManagedTensor *managed;
    PyObject *capsule;
    void *ptr;
    uint64_t nbytes;
    int32_t device_type;
    int32_t device_id;
} ExportObject;

static void export_release_impl(ExportObject *self) {
    if (self->managed != NULL) {
        ExoDLManagedTensor *m = self->managed;
        self->managed = NULL;
        if (m->deleter != NULL) {
            /* The producer's deleter may run Python code (ctypes, numpy, torch): never call it with an exception pending. */
            PyObject *pending = PyErr_GetRaisedException();
            m->deleter(m);
            PyErr_SetRaisedException(pending);
        }
    }
    Py_CLEAR(self->capsule);
    self->ptr = NULL;
}

static void export_dealloc(PyObject *o) {
    ExportObject *self = (ExportObject *)o;
    PyTypeObject *tp = Py_TYPE(o);
    export_release_impl(self);
    tp->tp_free(o);
    Py_DECREF(tp);
}

/* nbytes = ceil(bits*lanes/8) * prod(shape), overflow-checked. Returns -1 on overflow. */
static int dl_nbytes(const ExoDLTensor *t, uint64_t *out) {
    uint64_t elem_bits = (uint64_t)t->dtype.bits * (uint64_t)t->dtype.lanes;
    if (elem_bits == 0) return -1;
    uint64_t count = 1;
    for (int32_t i = 0; i < t->ndim; i++) {
        if (t->shape[i] < 0) return -1;
        if (__builtin_mul_overflow(count, (uint64_t)t->shape[i], &count)) return -1;
    }
    uint64_t bits_total;
    if (__builtin_mul_overflow(count, elem_bits, &bits_total)) return -1;
    *out = bits_total / 8 + ((bits_total % 8) != 0);
    return 0;
}

static int dl_is_c_contiguous(const ExoDLTensor *t) {
    if (t->strides == NULL) return 1;
    int64_t expected = 1;
    for (int32_t i = t->ndim - 1; i >= 0; i--) {
        if (t->shape[i] == 1) continue;
        if (t->strides[i] != expected) return 0;
        if (__builtin_mul_overflow(expected, t->shape[i], &expected)) return 0;
    }
    return 1;
}

static PyObject *export_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyObject *obj;
    static char *kwlist[] = {"obj", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &obj)) return NULL;

    int32_t method_type = -1, method_id = 0;
    int have_method = 0;
    if (PyObject_HasAttrString(obj, "__dlpack_device__")) {
        PyObject *dev = PyObject_CallMethod(obj, "__dlpack_device__", NULL);
        if (dev == NULL) return NULL;
        int ok = PyArg_ParseTuple(dev, "ii", &method_type, &method_id);
        Py_DECREF(dev);
        if (!ok) return NULL;
        have_method = 1;
    }
    PyObject *cap = PyObject_CallMethod(obj, "__dlpack__", NULL);
    if (cap == NULL) return NULL;
    if (!PyCapsule_IsValid(cap, "dltensor")) {
        Py_DECREF(cap);
        PyErr_SetString(PyExc_TypeError, "__dlpack__ did not return a 'dltensor' capsule (already consumed or versioned)");
        return NULL;
    }
    ExoDLManagedTensor *m = (ExoDLManagedTensor *)PyCapsule_GetPointer(cap, "dltensor");
    if (m == NULL) {
        Py_DECREF(cap);
        return NULL;
    }
    if (PyCapsule_SetName(cap, "used_dltensor") != 0) {
        Py_DECREF(cap);
        return NULL;
    }
    ExportObject *self = (ExportObject *)type->tp_alloc(type, 0);
    if (self == NULL) {
        PyObject *pending = PyErr_GetRaisedException();
        if (m->deleter) m->deleter(m);
        PyErr_SetRaisedException(pending);
        Py_DECREF(cap);
        return NULL;
    }
    self->managed = m;
    self->capsule = cap;
    const ExoDLTensor *t = &m->dl_tensor;
    int32_t cap_type = t->device.device_type;
    /* MLX reports the real device only via __dlpack_device__ (its capsule always says CPU). A GPU device in the capsule must agree. */
    if (have_method && cap_type != EXO_DL_CPU && cap_type != method_type) {
        PyErr_Format(PyExc_ValueError, "DLPack device mismatch: capsule %d vs __dlpack_device__ %d", (int)cap_type, (int)method_type);
        goto fail;
    }
    self->device_type = have_method ? method_type : cap_type;
    self->device_id = have_method ? method_id : t->device.device_id;
    if (t->ndim < 0 || (t->ndim > 0 && t->shape == NULL)) {
        PyErr_SetString(PyExc_ValueError, "DLPack tensor has an invalid shape");
        goto fail;
    }
    if (t->dtype.lanes == 0 || t->dtype.bits == 0) {
        PyErr_SetString(PyExc_ValueError, "DLPack tensor has an invalid dtype");
        goto fail;
    }
    if (dl_nbytes(t, &self->nbytes) != 0) {
        PyErr_SetString(PyExc_OverflowError, "DLPack tensor byte count overflows");
        goto fail;
    }
    if (!dl_is_c_contiguous(t)) {
        PyErr_SetString(PyExc_ValueError, "DLPack tensor is not C-contiguous; materialize it first");
        goto fail;
    }
    if (self->nbytes > 0 && t->data == NULL) {
        PyErr_SetString(PyExc_ValueError, "DLPack tensor has a null data pointer");
        goto fail;
    }
    self->ptr = (void *)((char *)t->data + t->byte_offset);
    return (PyObject *)self;
fail:
    Py_DECREF(self);
    return NULL;
}

static PyObject *export_get_ptr(PyObject *o, void *c) { (void)c; return PyLong_FromVoidPtr(((ExportObject *)o)->ptr); }
static PyObject *export_get_nbytes(PyObject *o, void *c) { (void)c; return PyLong_FromUnsignedLongLong(((ExportObject *)o)->nbytes); }
static PyObject *export_get_device(PyObject *o, void *c) {
    (void)c;
    ExportObject *e = (ExportObject *)o;
    return Py_BuildValue("ii", (int)e->device_type, (int)e->device_id);
}
static PyObject *export_release(PyObject *o, PyObject *unused) {
    (void)unused;
    export_release_impl((ExportObject *)o);
    Py_RETURN_NONE;
}
static PyObject *export_get_released(PyObject *o, void *c) { (void)c; return PyBool_FromLong(((ExportObject *)o)->managed == NULL); }

static PyGetSetDef export_getset[] = {
    {"ptr", export_get_ptr, NULL, "address of the first byte (data + byte_offset)", NULL},
    {"nbytes", export_get_nbytes, NULL, "payload size in bytes", NULL},
    {"device", export_get_device, NULL, "(DLPack device type, device id)", NULL},
    {"released", export_get_released, NULL, "True once the producer's deleter has run", NULL},
    {NULL, NULL, NULL, NULL, NULL}};
static PyMethodDef export_methods[] = {{"release", export_release, METH_NOARGS, "run the DLPack deleter (idempotent)"}, {NULL, NULL, 0, NULL}};
static PyType_Slot export_slots[] = {{Py_tp_new, export_new},
                                     {Py_tp_dealloc, export_dealloc},
                                     {Py_tp_getset, export_getset},
                                     {Py_tp_methods, export_methods},
                                     {0, NULL}};
static PyType_Spec export_spec = {"exo_tbccl._native.Export", sizeof(ExportObject), 0, Py_TPFLAGS_DEFAULT, export_slots};
static PyTypeObject *ExportType = NULL;

/* ------------------------------------------------------------------ Work */

typedef struct {
    PyObject_HEAD
    tbcclWork_t work;
} WorkObject;
static PyTypeObject *WorkType = NULL;

static PyObject *work_wrap(tbcclWork_t w) {
    WorkObject *o = PyObject_New(WorkObject, WorkType);
    if (o == NULL) {
        tbcclWorkDestroy(w);
        return NULL;
    }
    o->work = w;
    return (PyObject *)o;
}

static void work_dealloc(PyObject *o) {
    WorkObject *self = (WorkObject *)o;
    PyTypeObject *tp = Py_TYPE(o);
    if (self->work != NULL) tbcclWorkDestroy(self->work); /* drops the handle only; never cancels, never releases buffer obligations */
    PyObject_Free(o);
    Py_DECREF(tp);
}

static int work_check(WorkObject *w) {
    if (w->work == NULL) {
        PyErr_SetString(PyExc_ValueError, "Work is closed");
        return -1;
    }
    return 0;
}

/* wait(timeout_ms=None) -> (done, operation_result). The API status is separate from the operation result. */
static PyObject *work_wait(PyObject *o, PyObject *args) {
    WorkObject *self = (WorkObject *)o;
    PyObject *timeout = Py_None;
    if (!PyArg_ParseTuple(args, "|O", &timeout)) return NULL;
    if (work_check(self) != 0) return NULL;
    tbcclResult_t api, op = TBCCL_SUCCESS;
    int32_t done = 0;
    tbcclWork_t w = self->work;
    if (timeout == Py_None) {
        Py_BEGIN_ALLOW_THREADS api = tbcclWorkWait(w, &op);
        done = 1;
        Py_END_ALLOW_THREADS
    } else {
        unsigned long long ms = PyLong_AsUnsignedLongLong(timeout);
        if (PyErr_Occurred()) return NULL;
        Py_BEGIN_ALLOW_THREADS api = tbcclWorkWaitFor(w, (uint64_t)ms, &done, &op);
        Py_END_ALLOW_THREADS
    }
    if (api != TBCCL_SUCCESS) {
        raise_result(api, "work.wait", NULL);
        return NULL;
    }
    return Py_BuildValue("ii", (int)done, (int)op);
}

static PyObject *work_test(PyObject *o, PyObject *unused) {
    (void)unused;
    WorkObject *self = (WorkObject *)o;
    if (work_check(self) != 0) return NULL;
    int32_t done = 0;
    tbcclResult_t op = TBCCL_SUCCESS;
    tbcclResult_t api = tbcclWorkTest(self->work, &done, &op);
    if (api != TBCCL_SUCCESS) {
        raise_result(api, "work.test", NULL);
        return NULL;
    }
    return Py_BuildValue("ii", (int)done, (int)op);
}

static PyObject *work_error_string(PyObject *o, PyObject *unused) {
    (void)unused;
    WorkObject *self = (WorkObject *)o;
    if (work_check(self) != 0) return NULL;
    size_t required = 0;
    tbcclResult_t api = tbcclWorkGetErrorString(self->work, NULL, 0, &required);
    if (api != TBCCL_SUCCESS) {
        raise_result(api, "work.error_string", NULL);
        return NULL;
    }
    if (required <= 1) return PyUnicode_FromString("");
    char *buf = (char *)PyMem_Malloc(required);
    if (buf == NULL) return PyErr_NoMemory();
    api = tbcclWorkGetErrorString(self->work, buf, required, &required);
    PyObject *res = (api == TBCCL_SUCCESS) ? PyUnicode_DecodeUTF8(buf, (Py_ssize_t)strlen(buf), "replace") : NULL;
    if (api != TBCCL_SUCCESS) raise_result(api, "work.error_string", NULL);
    PyMem_Free(buf);
    return res;
}

static PyObject *work_close(PyObject *o, PyObject *unused) {
    (void)unused;
    WorkObject *self = (WorkObject *)o;
    if (self->work != NULL) {
        tbcclWork_t w = self->work;
        self->work = NULL;
        tbcclWorkDestroy(w);
    }
    Py_RETURN_NONE;
}

static PyMethodDef work_methods[] = {
    {"wait", work_wait, METH_VARARGS, "wait(timeout_ms=None) -> (done, operation_result); releases the GIL"},
    {"test", work_test, METH_NOARGS, "non-blocking (done, operation_result)"},
    {"error_string", work_error_string, METH_NOARGS, "terminal error text"},
    {"close", work_close, METH_NOARGS, "drop the handle (does not cancel the operation)"},
    {NULL, NULL, 0, NULL}};
static PyType_Slot work_slots[] = {{Py_tp_dealloc, work_dealloc}, {Py_tp_methods, work_methods}, {0, NULL}};
static PyType_Spec work_spec = {"exo_tbccl._native.Work", sizeof(WorkObject), 0, Py_TPFLAGS_DEFAULT, work_slots};

/* ------------------------------------------------------------------ Comm */

typedef struct {
    PyObject_HEAD
    tbcclComm_t comm;
} CommObject;
static PyTypeObject *CommType = NULL;

static int comm_check(CommObject *c) {
    if (c->comm == NULL) {
        PyErr_SetString(PyExc_ValueError, "communicator is closed");
        return -1;
    }
    return 0;
}

static void comm_dealloc(PyObject *o) {
    CommObject *self = (CommObject *)o;
    PyTypeObject *tp = Py_TYPE(o);
    if (self->comm != NULL) {
        tbcclComm_t c = self->comm;
        self->comm = NULL;
        Py_BEGIN_ALLOW_THREADS tbcclCommDestroy(c);
        Py_END_ALLOW_THREADS
    }
    PyObject_Free(o);
    Py_DECREF(tp);
}

static int fill_buffer(tbcclBuffer *b, unsigned long long ptr, unsigned long long nbytes, int kind, int device) {
    memset(b, 0, sizeof(*b));
    b->struct_size = (uint32_t)sizeof(*b);
    b->memory_kind = kind;
    b->device_ordinal = device;
    b->data = (void *)(uintptr_t)ptr;
    b->bytes = nbytes;
    return 0;
}

static PyObject *comm_p2p(CommObject *self, PyObject *args, int is_send) {
    unsigned long long ptr, nbytes;
    int kind, device;
    unsigned int peer;
    if (!PyArg_ParseTuple(args, "KKiiI", &ptr, &nbytes, &kind, &device, &peer)) return NULL;
    if (comm_check(self) != 0) return NULL;
    tbcclBuffer buf;
    fill_buffer(&buf, ptr, nbytes, kind, device);
    tbcclWork_t w = NULL;
    tbcclResult_t r = is_send ? tbcclSend(self->comm, &buf, peer, NULL, &w) : tbcclRecv(self->comm, &buf, peer, NULL, &w);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, is_send ? "send" : "recv", NULL);
        return NULL;
    }
    return work_wrap(w);
}
static PyObject *comm_send(PyObject *o, PyObject *args) { return comm_p2p((CommObject *)o, args, 1); }
static PyObject *comm_recv(PyObject *o, PyObject *args) { return comm_p2p((CommObject *)o, args, 0); }

static PyObject *comm_all_gather(PyObject *o, PyObject *args) {
    CommObject *self = (CommObject *)o;
    unsigned long long sptr, rptr, sbytes, rbytes;
    int kind, device;
    if (!PyArg_ParseTuple(args, "KKKKii", &sptr, &sbytes, &rptr, &rbytes, &kind, &device)) return NULL;
    if (comm_check(self) != 0) return NULL;
    tbcclBuffer s, r;
    fill_buffer(&s, sptr, sbytes, kind, device);
    fill_buffer(&r, rptr, rbytes, kind, device);
    tbcclWork_t w = NULL;
    tbcclResult_t rc = tbcclAllGather(self->comm, &s, &r, NULL, &w);
    if (rc != TBCCL_SUCCESS) {
        raise_result(rc, "all_gather", NULL);
        return NULL;
    }
    return work_wrap(w);
}

static PyObject *comm_barrier(PyObject *o, PyObject *unused) {
    (void)unused;
    CommObject *self = (CommObject *)o;
    if (comm_check(self) != 0) return NULL;
    tbcclWork_t w = NULL;
    tbcclResult_t rc = tbcclBarrier(self->comm, &w);
    if (rc != TBCCL_SUCCESS) {
        raise_result(rc, "barrier", NULL);
        return NULL;
    }
    return work_wrap(w);
}

static PyObject *comm_rank_size(PyObject *o, PyObject *unused) {
    (void)unused;
    CommObject *self = (CommObject *)o;
    if (comm_check(self) != 0) return NULL;
    uint32_t rank = 0, size = 0;
    CHECK_API(tbcclCommGetRank(self->comm, &rank), "comm.rank");
    CHECK_API(tbcclCommGetSize(self->comm, &size), "comm.size");
    return Py_BuildValue("II", rank, size);
fail:
    return NULL;
}

static PyObject *comm_abort(PyObject *o, PyObject *args) {
    CommObject *self = (CommObject *)o;
    const char *reason = "aborted";
    if (!PyArg_ParseTuple(args, "|s", &reason)) return NULL;
    if (comm_check(self) != 0) return NULL;
    tbcclResult_t r = tbcclCommAbort(self->comm, reason);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "comm.abort", NULL);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *comm_is_aborted(PyObject *o, PyObject *unused) {
    (void)unused;
    CommObject *self = (CommObject *)o;
    if (comm_check(self) != 0) return NULL;
    int32_t a = 0;
    tbcclResult_t r = tbcclCommIsAborted(self->comm, &a);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "comm.is_aborted", NULL);
        return NULL;
    }
    return PyBool_FromLong(a != 0);
}

static PyObject *comm_abort_reason(PyObject *o, PyObject *unused) {
    (void)unused;
    CommObject *self = (CommObject *)o;
    if (comm_check(self) != 0) return NULL;
    size_t required = 0;
    tbcclResult_t r = tbcclCommGetAbortReason(self->comm, NULL, 0, &required);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "comm.abort_reason", NULL);
        return NULL;
    }
    if (required <= 1) return PyUnicode_FromString("");
    char *buf = (char *)PyMem_Malloc(required);
    if (buf == NULL) return PyErr_NoMemory();
    r = tbcclCommGetAbortReason(self->comm, buf, required, &required);
    PyObject *res = (r == TBCCL_SUCCESS) ? PyUnicode_DecodeUTF8(buf, (Py_ssize_t)strlen(buf), "replace") : NULL;
    if (r != TBCCL_SUCCESS) raise_result(r, "comm.abort_reason", NULL);
    PyMem_Free(buf);
    return res;
}

static PyObject *comm_capabilities(PyObject *o, PyObject *unused) {
    (void)unused;
    CommObject *self = (CommObject *)o;
    if (comm_check(self) != 0) return NULL;
    tbcclCapabilities caps;
    memset(&caps, 0, sizeof(caps));
    caps.struct_size = (uint32_t)sizeof(caps);
    tbcclResult_t r = tbcclCommGetCapabilities(self->comm, &caps);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "comm.capabilities", NULL);
        return NULL;
    }
    return Py_BuildValue("IKK", caps.memory_kind_mask, (unsigned long long)caps.effective_max_chunk,
                         (unsigned long long)caps.effective_alignment);
}

static PyObject *comm_close(PyObject *o, PyObject *unused) {
    (void)unused;
    CommObject *self = (CommObject *)o;
    if (self->comm != NULL) {
        tbcclComm_t c = self->comm;
        self->comm = NULL;
        tbcclResult_t r;
        Py_BEGIN_ALLOW_THREADS r = tbcclCommDestroy(c);
        Py_END_ALLOW_THREADS
        if (r != TBCCL_SUCCESS) {
            raise_result(r, "comm.close", NULL);
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

static PyMethodDef comm_methods[] = {
    {"send", comm_send, METH_VARARGS, "send(ptr, nbytes, memory_kind, device, peer) -> Work"},
    {"recv", comm_recv, METH_VARARGS, "recv(ptr, nbytes, memory_kind, device, peer) -> Work"},
    {"all_gather", comm_all_gather, METH_VARARGS, "all_gather(sptr, sbytes, rptr, rbytes, kind, device) -> Work"},
    {"barrier", comm_barrier, METH_NOARGS, "barrier() -> Work"},
    {"rank_size", comm_rank_size, METH_NOARGS, "(rank, world_size)"},
    {"abort", comm_abort, METH_VARARGS, "abort(reason)"},
    {"is_aborted", comm_is_aborted, METH_NOARGS, NULL},
    {"abort_reason", comm_abort_reason, METH_NOARGS, NULL},
    {"capabilities", comm_capabilities, METH_NOARGS, "(memory_kind_mask, max_chunk, alignment)"},
    {"close", comm_close, METH_NOARGS, "destroy the communicator (bounded; quiesce callers first)"},
    {NULL, NULL, 0, NULL}};
static PyType_Slot comm_slots[] = {{Py_tp_dealloc, comm_dealloc}, {Py_tp_methods, comm_methods}, {0, NULL}};
static PyType_Spec comm_spec = {"exo_tbccl._native.Comm", sizeof(CommObject), 0, Py_TPFLAGS_DEFAULT, comm_slots};

/* ------------------------------------------------------------------ Bootstrap */

typedef struct {
    PyObject_HEAD
    tbcclBootstrap_t bs;
} BootstrapObject;
static PyTypeObject *BootstrapType = NULL;

static void bootstrap_dealloc(PyObject *o) {
    BootstrapObject *self = (BootstrapObject *)o;
    PyTypeObject *tp = Py_TYPE(o);
    if (self->bs != NULL) tbcclBootstrapDestroy(self->bs);
    tp->tp_free(o);
    Py_DECREF(tp);
}

static PyObject *bootstrap_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"rank", "world_size", "unique_id", "bind_host", "advertise_host", "timeout_ms", NULL};
    unsigned int rank, world, timeout_ms = 0;
    Py_buffer uid;
    const char *bind_host = NULL, *adv_host = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "IIy*|zzI", kwlist, &rank, &world, &uid, &bind_host, &adv_host, &timeout_ms)) return NULL;
    if (uid.len != (Py_ssize_t)sizeof(tbcclUniqueId)) {
        PyBuffer_Release(&uid);
        PyErr_SetString(PyExc_ValueError, "unique_id must be exactly 16 bytes");
        return NULL;
    }
    tbcclUniqueId id;
    memcpy(id.bytes, uid.buf, sizeof(id.bytes));
    PyBuffer_Release(&uid);
    tbcclBootstrapOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.struct_size = (uint32_t)sizeof(opts);
    opts.bind_host = bind_host;
    opts.advertise_host = adv_host;
    opts.timeout_ms = timeout_ms;
    tbcclBootstrap_t bs = NULL;
    tbcclResult_t r = tbcclBootstrapBegin(rank, world, &id, &opts, &bs);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "bootstrap.begin", NULL);
        return NULL;
    }
    BootstrapObject *self = (BootstrapObject *)type->tp_alloc(type, 0);
    if (self == NULL) {
        tbcclBootstrapDestroy(bs);
        return NULL;
    }
    self->bs = bs;
    return (PyObject *)self;
}

static PyObject *bootstrap_endpoint(PyObject *o, PyObject *unused) {
    (void)unused;
    BootstrapObject *self = (BootstrapObject *)o;
    if (self->bs == NULL) {
        PyErr_SetString(PyExc_ValueError, "bootstrap is closed");
        return NULL;
    }
    tbcclEndpointBlob blob;
    memset(&blob, 0, sizeof(blob));
    blob.struct_size = (uint32_t)sizeof(blob);
    tbcclResult_t r = tbcclBootstrapGetEndpoint(self->bs, &blob);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "bootstrap.endpoint", NULL);
        return NULL;
    }
    return PyBytes_FromStringAndSize((const char *)&blob, (Py_ssize_t)sizeof(blob));
}

/* complete(blobs: bytes of world_size * 256) -> Comm. Blocking, GIL released. */
static PyObject *bootstrap_complete(PyObject *o, PyObject *args) {
    BootstrapObject *self = (BootstrapObject *)o;
    Py_buffer blobs;
    if (!PyArg_ParseTuple(args, "y*", &blobs)) return NULL;
    if (self->bs == NULL) {
        PyBuffer_Release(&blobs);
        PyErr_SetString(PyExc_ValueError, "bootstrap is closed");
        return NULL;
    }
    if (blobs.len % (Py_ssize_t)TBCCL_ENDPOINT_BLOB_SIZE != 0) {
        PyBuffer_Release(&blobs);
        PyErr_SetString(PyExc_ValueError, "blobs must be a multiple of 256 bytes");
        return NULL;
    }
    uint32_t count = (uint32_t)(blobs.len / (Py_ssize_t)TBCCL_ENDPOINT_BLOB_SIZE);
    tbcclEndpointBlob *copy = (tbcclEndpointBlob *)PyMem_Malloc((size_t)blobs.len > 0 ? (size_t)blobs.len : 1);
    if (copy == NULL) {
        PyBuffer_Release(&blobs);
        return PyErr_NoMemory();
    }
    memcpy(copy, blobs.buf, (size_t)blobs.len);
    PyBuffer_Release(&blobs);
    tbcclComm_t comm = NULL;
    tbcclResult_t r;
    tbcclBootstrap_t bs = self->bs;
    Py_BEGIN_ALLOW_THREADS r = tbcclBootstrapComplete(bs, copy, count, &comm);
    Py_END_ALLOW_THREADS
    PyMem_Free(copy);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "bootstrap.complete", NULL);
        return NULL;
    }
    CommObject *c = PyObject_New(CommObject, CommType);
    if (c == NULL) {
        tbcclCommDestroy(comm);
        return NULL;
    }
    c->comm = comm;
    return (PyObject *)c;
}

static PyObject *bootstrap_close(PyObject *o, PyObject *unused) {
    (void)unused;
    BootstrapObject *self = (BootstrapObject *)o;
    if (self->bs != NULL) {
        tbcclBootstrap_t b = self->bs;
        self->bs = NULL;
        tbcclBootstrapDestroy(b);
    }
    Py_RETURN_NONE;
}

static PyMethodDef bootstrap_methods[] = {
    {"endpoint", bootstrap_endpoint, METH_NOARGS, "this rank's 256-byte opaque endpoint blob"},
    {"complete", bootstrap_complete, METH_VARARGS, "complete(blobs) -> Comm (blobs in rank order); releases the GIL"},
    {"close", bootstrap_close, METH_NOARGS, "release the listeners (safe after complete)"},
    {NULL, NULL, 0, NULL}};
static PyType_Slot bootstrap_slots[] = {{Py_tp_new, bootstrap_new}, {Py_tp_dealloc, bootstrap_dealloc}, {Py_tp_methods, bootstrap_methods}, {0, NULL}};
static PyType_Spec bootstrap_spec = {"exo_tbccl._native.Bootstrap", sizeof(BootstrapObject), 0, Py_TPFLAGS_DEFAULT, bootstrap_slots};

/* ------------------------------------------------------------------ module functions */

static PyObject *m_set_error_factory(PyObject *self, PyObject *f) {
    (void)self;
    if (!PyCallable_Check(f)) {
        PyErr_SetString(PyExc_TypeError, "factory must be callable");
        return NULL;
    }
    Py_XSETREF(g_error_factory, Py_NewRef(f));
    Py_RETURN_NONE;
}

static PyObject *m_abi_version(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    uint32_t v = 0;
    tbcclResult_t r = tbcclGetAbiVersion(&v);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "abi_version", NULL);
        return NULL;
    }
    return PyLong_FromUnsignedLong(v);
}

static PyObject *m_package_version(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    uint32_t a = 0, b = 0, c = 0;
    tbcclResult_t r = tbcclGetPackageVersion(&a, &b, &c);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "package_version", NULL);
        return NULL;
    }
    return Py_BuildValue("III", a, b, c);
}

static PyObject *m_unique_id(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    tbcclUniqueId id;
    tbcclResult_t r = tbcclGetUniqueId(&id);
    if (r != TBCCL_SUCCESS) {
        raise_result(r, "unique_id", NULL);
        return NULL;
    }
    return PyBytes_FromStringAndSize((const char *)id.bytes, (Py_ssize_t)sizeof(id.bytes));
}

static PyObject *m_result_string(PyObject *self, PyObject *arg) {
    (void)self;
    long code = PyLong_AsLong(arg);
    if (code == -1 && PyErr_Occurred()) return NULL;
    return PyUnicode_FromString(tbcclGetResultString((tbcclResult_t)code));
}

/* register_cuda() -> result code (UNSUPPORTED in a host-only TBCCL install). */
static PyObject *m_register_cuda(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    return PyLong_FromLong((long)tbcclRegisterCudaSupport());
}

static PyMethodDef module_methods[] = {
    {"set_error_factory", m_set_error_factory, METH_O, "factory(code, op, detail) -> exception instance"},
    {"abi_version", m_abi_version, METH_NOARGS, NULL},
    {"package_version", m_package_version, METH_NOARGS, NULL},
    {"unique_id", m_unique_id, METH_NOARGS, NULL},
    {"result_string", m_result_string, METH_O, NULL},
    {"register_cuda", m_register_cuda, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static int module_exec(PyObject *m) {
    ExportType = (PyTypeObject *)PyType_FromSpec(&export_spec);
    WorkType = (PyTypeObject *)PyType_FromSpec(&work_spec);
    CommType = (PyTypeObject *)PyType_FromSpec(&comm_spec);
    BootstrapType = (PyTypeObject *)PyType_FromSpec(&bootstrap_spec);
    if (!ExportType || !WorkType || !CommType || !BootstrapType) return -1;
    if (PyModule_AddObjectRef(m, "Export", (PyObject *)ExportType) < 0) return -1;
    if (PyModule_AddObjectRef(m, "Work", (PyObject *)WorkType) < 0) return -1;
    if (PyModule_AddObjectRef(m, "Comm", (PyObject *)CommType) < 0) return -1;
    if (PyModule_AddObjectRef(m, "Bootstrap", (PyObject *)BootstrapType) < 0) return -1;
#define ADD(name) if (PyModule_AddIntConstant(m, #name, name) < 0) return -1
    ADD(TBCCL_SUCCESS); ADD(TBCCL_INVALID_ARGUMENT); ADD(TBCCL_UNSUPPORTED); ADD(TBCCL_RESOURCE_EXHAUSTED);
    ADD(TBCCL_ABORTED); ADD(TBCCL_TIMEOUT); ADD(TBCCL_PROTOCOL_MISMATCH); ADD(TBCCL_TRANSPORT_ERROR);
    ADD(TBCCL_INTERNAL_ERROR); ADD(TBCCL_DEVICE_ERROR);
    ADD(TBCCL_MEMORY_HOST); ADD(TBCCL_MEMORY_CUDA); ADD(TBCCL_MEMORY_METAL_SHARED);
    ADD(TBCCL_C_ABI_VERSION); ADD(TBCCL_ENDPOINT_BLOB_SIZE);
#undef ADD
    if (PyModule_AddIntConstant(m, "DL_CPU", EXO_DL_CPU) < 0) return -1;
    if (PyModule_AddIntConstant(m, "DL_CUDA", EXO_DL_CUDA) < 0) return -1;
    if (PyModule_AddIntConstant(m, "DL_CUDA_HOST", EXO_DL_CUDA_HOST) < 0) return -1;
    if (PyModule_AddIntConstant(m, "DL_METAL", EXO_DL_METAL) < 0) return -1;
    if (PyModule_AddIntConstant(m, "DL_CUDA_MANAGED", EXO_DL_CUDA_MANAGED) < 0) return -1;
    return 0;
}

static PyModuleDef_Slot module_slots[] = {{Py_mod_exec, module_exec}, {0, NULL}};
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "exo_tbccl._native", "TBCCL C ABI v1 binding and DLPack consumer", 0, module_methods,
                                        module_slots, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__native(void) { return PyModuleDef_Init(&module_def); }
