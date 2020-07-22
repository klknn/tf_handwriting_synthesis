#!bash
set -euo pipefail

# disable cuda (TODO: how to enable assertions in cuda?)
CUDA_VISIBLE_DEVICES=NoDevFiles

echo "=== install test deps ==="

pip install mypy pytype flake8

echo "=== type checking ==="

pytype .

mypy --ignore-missing-imports .

echo "=== lint ==="

flake8 . --show-source --statistics

echo "=== unit test ==="

python3 -m unittest discover -s tfsq -p "*_test.py"

echo "=== integration test ==="

python3 train.py --v=1 --num_epochs=1 --root testdata --download=false --log_interval=1
