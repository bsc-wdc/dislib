export ComputingUnits=4

runcompss \
    --debug \
    --pythonpath=$(pwd) \
    --python_interpreter=python3 \
    ./tests/tests.py 2>&1 | tee output.log

result=$(cat output.log | egrep "OK|FAILED")

echo "Tests result: ${result}"

if [[ $result =~ FAILED ]]; then 
    exit 1
fi
