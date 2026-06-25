"""
Minimal i18n layer for GUI strings.

The current UI target language is English.
"""

from typing import Any, Dict


class I18n:
  STRINGS: Dict[str, str] = {
    "warnings.load_data_first": "Please load data first in the Data Upload page.",
    "warnings.preprocess_first": "Please preprocess data first in the Preprocessing page.",
    "warnings.select_algorithms_first": "Please select algorithms first in the Algorithm Configuration page.",
    "warnings.configure_split_first": "Please configure Data Split first in step 2.",
    "warnings.results_missing": "No results are currently available.",
    "errors.operation_failed": "The operation failed. Please check parameters and try again.",
    "actions.retry": "Retry",
    "actions.reset": "Reset",
    "details.technical": "Technical details",
    "ui.mode.quick": "Quick Start",
    "ui.mode.advanced": "Advanced",
    "sidebar.workflow": "Main Workflow",
    "sidebar.tools": "Standalone Tools",
    "sidebar.status": "Status",
  }

  @classmethod
  def t(cls, key: str, **kwargs: Any) -> str:
    text = cls.STRINGS.get(key, key)
    if not kwargs:
      return text
    try:
      return text.format(**kwargs)
    except Exception:
      return text


def t(key: str, **kwargs: Any) -> str:
  """Shorthand translation helper."""
  return I18n.t(key, **kwargs)
