#!/bin/bash
set -euo pipefail

# disable cuda (TODO: how to enable assertions in cuda?)
CUDA_VISIBLE_DEVICES=NoDevFiles

echo "=== lint ==="

flake8 . --show-source --statistics

echo "=== type checking ==="

pytype .

mypy --ignore-missing-imports .

echo "=== unit test ==="

coverage run --omit "*_test.py" -m unittest discover -s tfsq -p "*_test.py" --verbose
coverage report -m

echo "=== integration test ==="

python3 train.py --v=1 --num_epochs=1 --root testdata --download=false --log_interval=1
