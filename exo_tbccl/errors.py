"""Exceptions built from TBCCL's structured result codes (never from message text)."""


class TbcclError(RuntimeError):
    code: int
    op: str
    detail: str
    rank: int | None
    peer: int | None

    def __init__(
        self,
        code: int,
        op: str,
        detail: str = "",
        *,
        rank: int | None = None,
        peer: int | None = None,
    ) -> None:
        self.code = code
        self.op = op
        self.detail = detail
        self.rank = rank
        self.peer = peer
        super().__init__(self._render())

    def _render(self) -> str:
        where = f"rank={self.rank} " if self.rank is not None else ""
        if self.peer is not None:
            where += f"peer={self.peer} "
        name = type(self).__name__
        tail = f": {self.detail}" if self.detail else ""
        return f"{name}: {where}op={self.op} code={self.code}{tail}"

    def with_context(self, *, rank: int | None, peer: int | None = None) -> "TbcclError":
        self.rank = rank if rank is not None else self.rank
        self.peer = peer if peer is not None else self.peer
        self.args = (self._render(),)
        return self


class TbcclInvalidArgumentError(TbcclError):
    pass


class TbcclUnsupportedError(TbcclError):
    pass


class TbcclResourceExhaustedError(TbcclError):
    pass


class TbcclAbortedError(TbcclError):
    pass


class TbcclTimeoutError(TbcclError):
    pass


class TbcclProtocolMismatchError(TbcclError):
    pass


class TbcclTransportError(TbcclError):
    pass


class TbcclDeviceError(TbcclError):
    pass


class TbcclInternalError(TbcclError):
    pass


_BY_CODE: dict[int, type[TbcclError]] = {
    1: TbcclInvalidArgumentError,
    2: TbcclUnsupportedError,
    3: TbcclResourceExhaustedError,
    4: TbcclAbortedError,
    5: TbcclTimeoutError,
    6: TbcclProtocolMismatchError,
    7: TbcclTransportError,
    8: TbcclInternalError,
    9: TbcclDeviceError,
}


def error_for(code: int, op: str, detail: str | None = None) -> TbcclError:
    return _BY_CODE.get(code, TbcclError)(code, op, detail or "")
