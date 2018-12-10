#!/bin/bash

root_path="$(dirname "$(readlink -f "$0")")"
cd ${root_path}
export PYTHONPATH=$PYTHONPATH:${root_path}

coverage3 run --source dislib tests/tests.py
coverage3 report -m

bash <(curl -s https://codecov.io/bash)
