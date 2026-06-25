#!/usr/bin/env bash
# Launch the SCRBenchmark Streamlit interface.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="${SCRIPT_DIR}/src/scrbenchmark/app.py"

cd "$SCRIPT_DIR"

if [ -n "${PYTHON_BIN:-}" ]; then
    :
elif [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
elif [ -x "${SCRIPT_DIR}/venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
    cat <<EOF
Streamlit is not available in the selected Python environment.

Create and activate a local environment with:
  python3 -m venv .venv
  source .venv/bin/activate

Then install the dependencies with:
  ${PYTHON_BIN} -m pip install -r requirements.txt
EOF
    exit 1
fi

echo "Launching SCRBenchmark via Streamlit..."
exec "$PYTHON_BIN" -m streamlit run "$APP_PATH" --server.port 8501 --server.headless false
