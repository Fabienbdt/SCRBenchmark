"""Load reproducible method specifications from ``methods/*.yaml``.

The registry is intentionally data-driven: adding a method should normally mean
adding one YAML spec with a declarative command template. A thin adapter remains
useful only when the author code cannot read the benchmark input or expose
labels/embeddings in a reusable file. The algorithmic core remains in the
author package or vendored source named by the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METHODS_DIR = REPO_ROOT / "methods"


@dataclass(frozen=True)
class MethodSpec:
    """One method or composed protocol known to SCRBenchmark."""

    name: str
    display_name: str
    family: str
    runner: Mapping[str, Any]
    source: Mapping[str, Any]
    core_contract: str
    core_status: str
    report: bool = True
    aliases: tuple[str, ...] = ()
    output: Mapping[str, Any] | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MethodSpec":
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("Method spec is missing required field 'name'.")
        runner = raw.get("runner") or {}
        if not isinstance(runner, Mapping) or not runner.get("kind"):
            raise ValueError(f"Method {name!r} is missing runner.kind.")
        source = raw.get("source") or {}
        if not isinstance(source, Mapping):
            source = {"kind": str(source)}
        aliases = tuple(str(item).strip() for item in raw.get("aliases", []) if str(item).strip())
        output = raw.get("output") or {}
        return cls(
            name=name,
            display_name=str(raw.get("display_name") or name),
            family=str(raw.get("family") or runner.get("kind") or "unknown"),
            runner=runner,
            source=source,
            core_contract=str(raw.get("core_contract") or "author_core_unchanged"),
            core_status=str(raw.get("core_status") or "unknown"),
            report=bool(raw.get("report", True)),
            aliases=aliases,
            output=output if isinstance(output, Mapping) else {},
            notes=str(raw.get("notes") or ""),
        )

    @property
    def runner_kind(self) -> str:
        return str(self.runner.get("kind"))

    @property
    def expected_file(self) -> str:
        output = self.output or {}
        return str(output.get("expected_file") or "results/analysis_results.csv")


def _read_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load method specs.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _iter_spec_files(methods_dir: Path) -> Iterable[Path]:
    if not methods_dir.exists():
        return []
    return sorted(
        path
        for path in methods_dir.rglob("*.yaml")
        if not path.name.startswith("_") and path.name != "template_method.yaml"
    )


def _iter_raw_specs(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("methods"), list):
        for item in payload["methods"]:
            if isinstance(item, Mapping):
                yield item
        return
    if isinstance(payload, Mapping) and payload.get("name"):
        yield payload


def load_method_specs(methods_dir: str | Path | None = None) -> dict[str, MethodSpec]:
    """Load all method specs keyed by canonical name and aliases."""

    root = Path(methods_dir).expanduser().resolve() if methods_dir else DEFAULT_METHODS_DIR
    specs: dict[str, MethodSpec] = {}
    alias_to_name: dict[str, str] = {}

    for path in _iter_spec_files(root):
        payload = _read_yaml(path)
        for raw in _iter_raw_specs(payload):
            spec = MethodSpec.from_mapping(raw)
            if spec.name in specs:
                raise ValueError(f"Duplicate method spec name: {spec.name}")
            specs[spec.name] = spec
            for alias in (spec.name, spec.display_name, *spec.aliases):
                key = alias.casefold()
                previous = alias_to_name.get(key)
                if previous and previous != spec.name:
                    raise ValueError(f"Alias {alias!r} maps to both {previous!r} and {spec.name!r}")
                alias_to_name[key] = spec.name

    for alias, canonical in alias_to_name.items():
        specs.setdefault(alias, specs[canonical])
    return specs


def get_method_spec(name: str, methods_dir: str | Path | None = None) -> MethodSpec | None:
    """Return a method spec by canonical name, display name, or alias."""

    specs = load_method_specs(methods_dir)
    return specs.get(str(name)) or specs.get(str(name).casefold())
