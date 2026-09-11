"""Keep the maintained onboarding links usable after documentation moves."""

from pathlib import Path
import re
from urllib.parse import unquote


def test_onboarding_local_links_resolve():
    root = Path(__file__).resolve().parents[2]
    documents = [root / "README.md", root / "CONTRIBUTING.md", *sorted((root / "docs/guide").glob("*.md"))]
    broken = []
    for document in documents:
        text = re.sub(r"```.*?```", "", document.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            path = unquote(target.split("#", 1)[0])
            if not (document.parent / path).exists():
                broken.append(f"{document.relative_to(root)} -> {target}")
    assert not broken, "Broken onboarding links:\n" + "\n".join(broken)
