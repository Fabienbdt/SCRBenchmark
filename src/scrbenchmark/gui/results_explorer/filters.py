"""Filters and selectors for Results Explorer."""

from typing import Dict, List

from .legacy import get_available_metrics
from .constants import FIGURE_CATEGORY_MAP


def filter_figure_keys(
  available_figures: List[str],
  search_text: str,
  categories: List[str],
  figure_descriptions: Dict[str, Dict[str, str]],
) -> List[str]:
  """Filter figure keys by text and selected categories."""
  filtered = []
  search_text = (search_text or "").strip().lower()
  selected_categories = set(categories or [])

  for key in available_figures:
    category = FIGURE_CATEGORY_MAP.get(key, "Other")
    if selected_categories and category not in selected_categories:
      continue

    if search_text:
      haystack = " ".join(
        [
          key,
          figure_descriptions.get(key, {}).get("name", ""),
          figure_descriptions.get(key, {}).get("description", ""),
          category,
        ]
      ).lower()
      if search_text not in haystack:
        continue

    filtered.append(key)

  return filtered
