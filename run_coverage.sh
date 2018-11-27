#!/bin/bash

# We make sure that ssh daemon is running
/etc/init.d/ssh start

coverage3 run --source dislib/dislib dislib/tests/tests.py
coverage3 report -m

bash <(curl -s https://codecov.io/bash)
