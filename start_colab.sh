#!/usr/bin/env bash
# Starts a Jupyter Notebook server configured for Google Colab local runtime.
# Run from the project root: ./start_colab.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "ERROR: No .venv found at $VENV"
  exit 1
fi

source "$VENV/bin/activate"

echo "Jupyter Notebook server starting..."
echo "In Colab: Connect -> Connect to a local runtime -> paste the URL below"
echo ""

jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --NotebookApp.disable_check_xsrf=True \
  --port=8888 \
  --no-browser

jupyter notebook list
  
