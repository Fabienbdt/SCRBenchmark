"""Include small runtime resources without bundling datasets or experiment outputs."""

from pathlib import Path
import shutil

from setuptools.command.build_py import build_py


RESOURCE_PATTERNS = {
    "data": ("README.md",),
    "data/report_sources": ("source_manifest.csv",),
    "data/stable_generalist": ("download_manifest.csv",),
    "scripts/setup": ("*.py", "*.md"),
    "methods": ("*.yaml", "*.md"),
    "protocols": ("*.yaml", "*.md"),
    "reproducibility": ("*.csv", "*.json", "*.yaml", "*.md"),
    "scripts/reproduction": ("*.py", "*.json", "*.yaml", "*.md"),
    "vendor/scraw_inductive": ("*.py", "*.json", "*.yaml", "*.md"),
    "vendor/scraw_dedicated": ("*.py", "*.json", "*.yaml", "*.md"),
    "vendor/stable_generalist_runners": ("*.py", "*.json", "*.yaml", "*.md"),
    "docs/guide": ("*.md",),
}


class BuildPy(build_py):
    """Keep the checkout layout inside the wheel for standalone report runners."""

    def run(self):
        super().run()
        source = Path(__file__).resolve().parents[2]
        target = Path(self.build_lib) / "scrbenchmark" / "_assets"
        # A repeated build must not retain resources removed from the source tree.
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for name in ("README.md", "CONTRIBUTING.md", "THIRD_PARTY.md", "CHANGELOG.md"):
            shutil.copy2(source / name, target / name)
        for directory, patterns in RESOURCE_PATTERNS.items():
            for pattern in patterns:
                for path in sorted((source / directory).rglob(pattern)):
                    if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(source).parts):
                        continue
                    destination = target / path.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        target = Path(self.build_lib) / "scrbenchmark" / "_assets"
        return outputs + [str(path) for path in target.rglob("*") if path.is_file()]
