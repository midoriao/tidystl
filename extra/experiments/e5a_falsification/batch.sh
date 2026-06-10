#!/usr/bin/env bash

set -euo pipefail

for b in native breach rtamt; do
  uv run python run.py --backend "$b"
done
