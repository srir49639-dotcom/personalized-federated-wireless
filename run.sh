#!/usr/bin/env bash
# ============================================================================
# Linux/macOS Shell Launcher for Personalized Federated Wireless Predictor
# ============================================================================

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "Starting Personalized Federated Wireless Application..."
python3 run.py || python run.py
