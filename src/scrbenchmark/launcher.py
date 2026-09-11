"""Console entry point for the installed Streamlit application."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Launch the GUI with the current interpreter and forward Streamlit options."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print('The GUI requires: python -m pip install "SCRBenchmark[gui]"', file=sys.stderr)
        return 1
    app = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]])
