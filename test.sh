#!bash
set -euo pipefail

echo "=== install test deps ==="

pip install mypy pytype

echo "=== type checking ==="

pytype .

mypy --ignore-missing-imports .

echo "=== integration test with mock data ==="

python3 train.py --v=1 --num_epochs=1 --batch_size=12 --lr 1e-4 --root testdata --download=false --log_interval=1
