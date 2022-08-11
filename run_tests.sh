#!/bin/bash -e

# Default process per worker
export ComputingUnits=1

declare -a tests_group=("test_lasso" 
                "test_array test_pca test_daura"
                "test_gm test_preproc test_decision_tree"
                "test_qr test_kmeans test_knn"
                "test_gridsearch test_tsqr test_linear_regression"
                "test_dbscan test_matmul test_als"
                "test_rf_classifier test_randomizedsearch test_data_utils test_kfold"
                "test_csvm test_rf_regressor test_utils test_rf_dataset"
                )

port=43001

for t in "${tests_group[@]}"
do
    port=$((port + 1))
    runcompss \
        --pythonpath=$(pwd) \
        --python_interpreter=python3 \
        --master_port=$port \
        ./tests/__main__.py $t &> >(tee output.log) &

    sleep 10
done

# Check the unittest output because PyCOMPSs exits with code 0 even if there
# are failed tests (the execution itself is successful)
result=$(cat output.log | egrep "OK|FAILED")

echo "Tests result: ${result}"

# If word Failed is in the results, exit 1 so the pull request fails
if [[ $result =~ FAILED ]]; then 
        exit 1
fi
