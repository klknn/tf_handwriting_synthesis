#!/bin/bash
set -euo pipefail

# disable cuda (TODO: how to enable assertions in cuda?)
CUDA_VISIBLE_DEVICES=NoDevFiles

echo "=== lint ==="

flake8 tfsq --show-source --statistics --ignore=E203,W503

black --check tfsq

echo "=== type checking ==="

pytype tfsq

mypy --ignore-missing-imports .

echo "=== unit test ==="

coverage run --omit "*_test.py" -m unittest discover -s tfsq -p "*_test.py" --verbose
coverage report -i

echo "=== integration test ==="

coverage run --append -m tfsq.train --v=1 --num_epochs=1 --root testdata --download=false --log_interval=1 --hidden_size=4 --resume=testdata/checkpoint/ep0_it1
coverage report -i
