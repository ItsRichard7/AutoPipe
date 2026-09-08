"""
reporter -- result reporting and reproducibility.

Public API:
    ReportGenerator             generates HTML (always) and optional PDF reports with the
                                comparison table, metric evolution chart, winning pipeline
                                config (re-runnable), and highlighted anomaly images
    ComparativeReportGenerator  extends the above with the leave-one-dataset-out report:
                                cold versus warm start, its secondary controls and the
                                robustness of the censoring rule
    redactar_veredicto          states the primary contrast against the committed
                                success criterion, favourable or not
"""

from autopipe.reporter.comparison import ComparativeReportGenerator, redactar_veredicto
from autopipe.reporter.report import ReportGenerator

__all__ = ["ComparativeReportGenerator", "ReportGenerator", "redactar_veredicto"]
