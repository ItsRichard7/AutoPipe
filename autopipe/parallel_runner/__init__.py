"""
parallel_runner -- concurrent pipeline evaluation.

Public API:
    ParallelRunner  evaluates multiple pipelines in parallel using joblib (loky backend)
"""

from autopipe.parallel_runner.runner import ParallelRunner

__all__ = ["ParallelRunner"]