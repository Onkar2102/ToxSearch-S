"""MPI parallel evolutionary execution (master–worker)."""

from .master_worker import run, resolve_parallel_merge_k

__all__ = ["run", "resolve_parallel_merge_k"]
