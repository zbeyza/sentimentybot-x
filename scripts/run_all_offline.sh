#!/usr/bin/env bash
set -euo pipefail

python -m src.main --train --evaluate --predict --analysis
