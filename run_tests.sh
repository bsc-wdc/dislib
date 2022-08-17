#!/bin/bash -e

# Default process per worker
export ComputingUnits=8

declare -a tests_group=("test_lasso" 
                "test_array test_pca test_daura"
                "test_gm test_preproc test_decision_tree"
                "test_qr test_kmeans test_knn"
                "test_gridsearch test_tsqr test_linear_regression"
                "test_dbscan test_matmul test_als"
                "test_rf_classifier test_randomizedsearch test_data_utils test_kfold"
                "test_csvm test_rf_regressor test_utils test_rf_dataset"
                )

declare -a pids

port=43000
workerid=0

for t in "${tests_group[@]}"
do
    
    nextport=$((port + 1))
    
    sed "s/<MinPort>43001<\/MinPort>/<MinPort>$port<\/MinPort>/g" /opt/COMPSs/Runtime/configuration/xml/resources/default_resources.xml > /tmp/resources-$port.xml
    sed -i "s/<MaxPort>43002<\/MaxPort>/<MaxPort>$nextport<\/MaxPort>/g" /tmp/resources-$port.xml

    runcompss \
        --pythonpath=$(pwd) \
        --python_interpreter=python3 \
        --resources=/tmp/resources-$port.xml \
        --master_port=$port \
        ./tests/__main__.py $t -id $workerid &> >(tee output.log) &

    pids+=($!)

    workerid=$((workerid + 1))

    port=$((port + 2))
    sleep 10
done

wait ${pids[@]}

# Check the unittest output because PyCOMPSs exits with code 0 even if there
# are failed tests (the execution itself is successful)
result=$(cat output.log | egrep "OK|FAILED")

echo "Tests result: ${result}"


# If word Failed is in the results, exit 1 so the pull request fails
if [[ $result =~ FAILED ]]; then 
        exit 1
fi
