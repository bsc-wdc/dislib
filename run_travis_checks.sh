root_path="$(dirname "$(readlink -f "$0")")"
cd ${root_path}

./run_tests.sh
./run_coverage.sh
./run_style.sh
