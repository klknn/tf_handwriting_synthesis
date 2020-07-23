#!/bin/bash

set -euo pipefail

cd docs

sphinx-apidoc -f -e -o . ../tfsq "../tfsq/*_test.py"

make SPHINXOPTS="-W" html
