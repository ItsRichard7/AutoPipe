"""
search_space -- canonical, versioned search spaces shared across datasets.

Public API:
    TABULAR_SPACE_VERSION        identifier of the frozen tabular space
    TABULAR_SPACE_STEPS          step names, in canonical order
    build_tabular_search_space   fresh instances of the canonical tabular space
    validate_space_version       rejects configurations from another version
    describe_tabular_search_space  step/index enumeration, for docs and reports
"""

from autopipe.search_space.tabular import (
    TABULAR_SPACE_STEPS,
    TABULAR_SPACE_VERSION,
    build_tabular_search_space,
    describe_tabular_search_space,
    validate_space_version,
)

__all__ = [
    "TABULAR_SPACE_VERSION",
    "TABULAR_SPACE_STEPS",
    "build_tabular_search_space",
    "validate_space_version",
    "describe_tabular_search_space",
]
