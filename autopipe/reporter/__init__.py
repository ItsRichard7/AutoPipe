"""
reporter -- result reporting and reproducibility.

Public API:
    ReportGenerator  generates HTML (always) and optional PDF reports with the
                     comparison table, metric evolution chart, winning pipeline
                     config (re-runnable), and highlighted anomaly images
"""

from autopipe.reporter.report import ReportGenerator

__all__ = ["ReportGenerator"]