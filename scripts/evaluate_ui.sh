#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
port=8765
python3 -B -m evaluation_ui --port "$port"
