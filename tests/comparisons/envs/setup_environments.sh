#!/usr/bin/env bash
# Script to create conda environments for original algorithm implementations
# Uses exact Python versions specified in original READMEs
#
# Usage:
#   ./setup_environments.sh [--replace] [algorithm_name]
#   ./setup_environments.sh [--replace] all
#
# Examples:
#   ./setup_environments.sh scdeepcluster
#   ./setup_environments.sh scmae
#   ./setup_environments.sh --replace scmae
#   ./setup_environments.sh all

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENVS_DIR="${SCRIPT_DIR}"
REPLACE_EXISTING=0

get_python_version() {
    local algo=$1
    case "$algo" in
        scdeepcluster) echo "3.9" ;;   # Not specified, using 3.9 for PyTorch 1.8 compatibility
        sccdcg) echo "3.7" ;;          # Explicitly requires Python 3.7
        scmae) echo "3.10" ;;          # Explicitly requires Python 3.10
        scname) echo "3.8" ;;          # Requires Python >= 3.8
        *) echo "" ;;
    esac
}

create_env() {
    local algo=$1
    local env_name="env_${algo}"
    local req_file="${ENVS_DIR}/requirements_${algo}.txt"
    local python_version=$(get_python_version "$algo")

    if [ ! -f "$req_file" ]; then
        echo "Error: Requirements file not found: $req_file"
        return 1
    fi

    if [ -z "$python_version" ]; then
        echo "Error: Python version not defined for $algo"
        return 1
    fi

    echo "============================================"
    echo "Creating conda environment for: $algo"
    echo "Python version: $python_version"
    echo "Environment name: $env_name"
    echo "============================================"

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found. Please install Anaconda or Miniconda."
        return 1
    fi

    # Preserve an existing environment unless replacement was explicitly requested.
    if conda env list | grep -q "^${env_name} "; then
        if [ "$REPLACE_EXISTING" -ne 1 ]; then
            echo "Error: Conda environment '${env_name}' already exists."
            echo "Re-run with --replace to remove and recreate that environment."
            return 1
        fi
        echo "Removing existing environment '${env_name}' (--replace requested)..."
        conda env remove -n "$env_name" -y
    fi

    # Create new environment with specific Python version
    echo "Creating conda environment with Python $python_version..."
    conda create -n "$env_name" python="$python_version" -y

    # Activate and install requirements
    echo "Installing requirements..."

    # Use conda run to install in the environment
    conda run -n "$env_name" pip install --upgrade pip
    conda run -n "$env_name" pip install -r "$req_file"

    # Install pytest for running tests
    conda run -n "$env_name" pip install pytest pytest-timeout

    echo ""
    echo "Environment for $algo created successfully!"
    echo "To activate: conda activate $env_name"
    echo ""
}

print_summary() {
    echo ""
    echo "============================================"
    echo "ALGORITHM PYTHON VERSION REQUIREMENTS"
    echo "(from original authors' READMEs)"
    echo "============================================"
    echo ""
    echo "| Algorithm     | Python | Key Dependencies           |"
    echo "|---------------|--------|----------------------------|"
    echo "| scDeepCluster | 3.9    | PyTorch 1.8, Scanpy 1.7    |"
    echo "| scCDCG        | 3.7    | PyTorch 1.12, Keras 2.4.3  |"
    echo "| scMAE         | 3.10   | PyTorch Lightning 2.0+     |"
    echo "| scNAME        | 3.8    | TensorFlow 2.2+, Keras 2.3+|"
    echo ""
}

# Parse the optional destructive replacement flag.
if [ "${1:-}" == "--replace" ]; then
    REPLACE_EXISTING=1
    shift
fi

if [ $# -ne 1 ]; then
    echo "Usage: ./setup_environments.sh [--replace] [algorithm_name|all|summary]"
    echo "Available algorithms: scdeepcluster, sccdcg, scmae, scname"
    echo "Use 'summary' to see version requirements"
    echo "Use '--replace' only when an existing matching environment may be removed."
    exit 1
fi

ALGO=$1

if [ "$ALGO" == "summary" ]; then
    print_summary
elif [ "$ALGO" == "all" ]; then
    print_summary
    echo "Creating all environments..."
    echo ""
    failed=0
    for algo in scdeepcluster sccdcg scmae scname; do
        create_env "$algo" || failed=1
    done
    if [ "$failed" -ne 0 ]; then
        echo "One or more environments could not be created."
        exit 1
    fi
    echo "============================================"
    echo "All environments created!"
    echo "============================================"
else
    create_env "$ALGO" || exit $?
fi

echo ""
echo "To activate an environment, run:"
echo "  conda activate env_<algorithm>"
echo ""
echo "To run comparison tests:"
echo "  conda activate env_<algorithm>"
echo "  cd tests/comparisons"
echo "  python compare_<algorithm>.py"
