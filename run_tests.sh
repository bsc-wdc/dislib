export ComputingUnits=4
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export MPI_HOME=/usr/lib64/openmpi
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib
export PATH=$PATH:/opt/COMPSs/Runtime/scripts/user
export CLASSPATH=$CLASSPATH:/opt/COMPSs/Runtime/compss-engine.jar
export PATH=$PATH:/opt/COMPSs/Bindings/c/bin

/etc/init.d/ssh start

coverage run --source dislib tests/tests.py
coverage report -m

runcompss \
    --debug \
    ./tests/tests.py 2>&1 | tee output.log

result=$(cat output.log | egrep "OK|FAILED")

echo "Tests result: ${result}"

if [[ $result =~ FAILED ]]; then 
    exit 1
fi
