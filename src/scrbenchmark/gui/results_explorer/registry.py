"""Declarative figure registry for Results Explorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class FigureInfo:
  key: str
  name: str
  category: str
  description: str = ""
  requires_conditions: bool = True
  requires_algorithms: bool = True
  tags: tuple[str, ...] = field(default_factory=tuple)


class FigureRegistry:
  _infos: Dict[str, FigureInfo] = {}
  _renderers: Dict[str, Callable[..., Any]] = {}

  @classmethod
  def register(cls, info: FigureInfo):
    def _decorator(renderer: Callable[..., Any]):
      cls._infos[info.key] = info
      cls._renderers[info.key] = renderer
      return renderer

    return _decorator

  @classmethod
  def info(cls, key: str) -> Optional[FigureInfo]:
    return cls._infos.get(key)

  @classmethod
  def keys(cls) -> List[str]:
    return list(cls._infos.keys())

  @classmethod
  def list_infos(cls) -> List[FigureInfo]:
    return [cls._infos[k] for k in cls.keys()]

  @classmethod
  def filter_available(
    cls,
    *,
    selected_algorithms: Optional[Iterable[str]] = None,
    selected_conditions: Optional[Iterable[str]] = None,
    search_text: str = "",
    categories: Optional[Iterable[str]] = None,
  ) -> List[FigureInfo]:
    algos = list(selected_algorithms or [])
    conds = list(selected_conditions or [])
    search = (search_text or "").strip().lower()
    cat_set = set(categories or [])

    available: List[FigureInfo] = []
    for info in cls.list_infos():
      if info.requires_algorithms and not algos:
        continue
      if info.requires_conditions and not conds:
        continue
      if cat_set and info.category not in cat_set:
        continue

      if search:
        haystack = " ".join([info.key, info.name, info.description, info.category, *info.tags]).lower()
        if search not in haystack:
          continue

      available.append(info)

    return available

  @classmethod
  def render(cls, key: str, **context: Any) -> Any:
    if key not in cls._renderers:
      raise KeyError(f"Unknown figure key: {key}")
    return cls._renderers[key](**context)


def debug_registry_state(
  selected_algorithms: Optional[Iterable[str]],
  selected_conditions: Optional[Iterable[str]],
  search_text: str,
  categories: Optional[Iterable[str]],
) -> Dict[str, List[str]]:
  """Return visible/hidden keys for debugging availability filtering."""
  algos = list(selected_algorithms or [])
  conds = list(selected_conditions or [])
  search = (search_text or "").strip().lower()
  cat_set = set(categories or [])

  visible: List[str] = []
  hidden: List[str] = []

  for info in FigureRegistry.list_infos():
    reasons = []
    if info.requires_algorithms and not algos:
      reasons.append("requires_algorithms")
    if info.requires_conditions and not conds:
      reasons.append("requires_conditions")
    if cat_set and info.category not in cat_set:
      reasons.append("category")
    if search:
      haystack = " ".join([info.key, info.name, info.description, info.category, *info.tags]).lower()
      if search not in haystack:
        reasons.append("search")

    if reasons:
      hidden.append(f"{info.key}: {', '.join(reasons)}")
    else:
      visible.append(info.key)

  return {"visible": visible, "hidden": hidden}
