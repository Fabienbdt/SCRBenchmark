"""Locate read-only runtime assets in a source checkout or an installed wheel.

User datasets and generated outputs must use explicit paths or the working
directory; they must never be written into this asset directory.
"""

from pathlib import Path


def resource_root() -> Path:
    """Return the packaged assets, or their original location for editable installs."""
    package_dir = Path(__file__).resolve().parent
    packaged = package_dir / "_assets"
    if packaged.is_dir():
        return packaged
    checkout = package_dir.parents[1]
    if (checkout / "methods" / "report_methods.yaml").is_file():
        return checkout
    raise FileNotFoundError(
        "SCRBenchmark runtime assets are missing. Reinstall the package, or run "
        "'python -m pip install -e .[gui]' from a complete source checkout."
    )
