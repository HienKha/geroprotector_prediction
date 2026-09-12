"""Fail-closed classification of resource exhaustion during locked model search."""

from __future__ import annotations


def is_resource_exhaustion(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, MemoryError):
            return True
        message = str(current).lower()
        if any(
            token in message
            for token in (
                "out of memory",
                "cannot allocate memory",
                "std::bad_alloc",
                "cuda_error_out_of_memory",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def abort_on_resource_exhaustion(exc: BaseException, *, stage: str) -> None:
    """Never let hardware limits silently shrink a prespecified candidate portfolio."""

    if is_resource_exhaustion(exc):
        raise RuntimeError(
            f"{stage} exhausted memory; aborting the locked run rather than "
            "treating this as a failed scientific candidate"
        ) from exc
