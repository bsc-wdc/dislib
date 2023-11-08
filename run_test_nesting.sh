#!/bin/bash

base_app_dir="$HOME/Documentos/GitHub/dislib/tests_nesting/"
COMPSs_log_folder="/tmp/COMPSsWorker01"
target_log_folder="$HOME/Documentos/GitHub/dislib/"
retry_num=1

echo $base_app_dir
echo $COMPSs_log_folder
echo $target_log_folder

AGENT_PIDS=""
exit_value=0
expected_time="60"
NUM_RETRIES="50"
app_name="Nesting_Tests"

  # Traps and Handlers
function kill_agents() {
  for pid in ${AGENT_PIDS}; do
    kill -SIGINT ${pid} 2>/dev/null
  done
}
trap kill_agents EXIT

#sed -i '/<InstallDir>/c<InstallDir>'"${COMPSS_HOME}"'<\/InstallDir>' "${base_app_dir}"/project.xml

mkdir -p /tmp/COMPSsWorker01/

echo ""
echo "*** RUNNING AGENTS TESTS ON DISLIB ***"
log_dir="${COMPSs_log_folder}/${app_name}_0${retry_num}/"
mkdir -p "${log_dir}"
output_log="${log_dir}test.outputlog"
error_log="${log_dir}test.errorlog"
touch "${output_log}"
touch "${error_log}"

port_offset=100

for file in "${base_app_dir}"test_*; do

  corresponding_file=$(echo "${file}" | cut -d '/' -f8)
  corresponding_file=$(echo "${corresponding_file}" | cut -d '.' -f1)

  log_dir="${COMPSs_log_folder}/${app_name}_0${retry_num}/"
  mkdir -p "${log_dir}"
  output_log="${log_dir}test.outputlog"
  error_log="${log_dir}test.errorlog"
  touch "${output_log}"
  touch "${error_log}"

  # Starting agent
  agent1_log_dir="${log_dir}/agent1/"
  mkdir -p "${agent1_log_dir}"
  agent1_output_log="${log_dir}agent1.outputlog"
  agent1_error_log="${log_dir}agent1.errorlog"

  rest_port=$(( 46000 + port_offset + 1))
  comm_port=$(( 46000 + port_offset + 2))
  which compss_agent_start
  compss_agent_start \
    --hostname="COMPSsWorker01" \
    --classpath="${base_app_dir}" \
    --log_dir="${agent1_log_dir}" \
    --rest_port="${rest_port}" \
    --comm_port="${comm_port}" \
    --pythonpath="${base_app_dir}" \
    --python_interpreter="python3"\
      1>"${agent1_output_log}" 2>"${agent1_error_log}" &

  agent_pid="$!"

  AGENT_PIDS="${AGENT_PIDS} ${agent_pid}"
  retries="${NUM_RETRIES}"
  echo "testing first agent"
  curl -XGET http://127.0.0.1:${rest_port}/COMPSs/test 1>/dev/null 2>/dev/null
  ev=$?

  while [ "$ev" != "0" ] && [ "${retries}" -gt "0" ]; do
    echo "testing agent on port ${rest_port}"
    sleep 2s
    retries=$((retries - 1 ))
    curl -XGET http://127.0.0.1:${rest_port}/COMPSs/test 1>/dev/null 2>/dev/null
    ev=$?
  done
  echo "TEST invoked"
  RESULT=$(grep "test invoked" "${agent1_output_log}")
  if [ -z "${RESULT}" ]; then
      echo "Agent failed to start" > >(tee -a "${error_log}")
      exit 1
  fi
  echo "Agent started" > >(tee -a "${output_log}")
  sleep 2s

  # Invoking DemoFunction method
  "${COMPSS_HOME}/Runtime/scripts/user/compss_agent_call_operation" \
    --lang="PYTHON" \
    --master_node="127.0.0.1" \
    --master_port="${rest_port}" \
    --method_name="main" \
    --stop \
    "${corresponding_file}" > >(tee -a "${output_log}") 2> >(tee -a "${error_log}")
    ev=$?
    if [ "$ev" != "0" ]; then
    echo "Could not invoke main method." > >(tee -a "${error_log}")
    exit $ev
  fi
  echo "main function invoked" > >(tee -a "${output_log}")

  retries="3"
  while [ ! -f "${agent1_log_dir}/jobs/job1_NEW.out" ] && [ "${retries}" -gt "0" ]; do
    sleep 2s
    retries=$((retries - 1 ))
  done
  if [ ! -f "${agent1_log_dir}/jobs/job1_NEW.out" ]; then
    echo "Could not invoke main method." > >(tee -a "${error_log}")
    exit 1
  fi

  wait ${AGENT_PIDS}

  if [ ! -f "${agent1_log_dir}/jobs/job2_NEW.out" ]; then
    echo "Could not invoke nested method." > >(tee -a "${error_log}")
    exit 1
  fi

  job1_end=$(grep "Result tests" "${agent1_log_dir}/jobs/job1_NEW.out")
  job1_end_value=$(echo "${job1_end}" | cut -d ' ' -f3)

  if [ ! "${job1_end_value}" == "Passed" ]; then
    echo "Unexpected integer value obtained from the test. Expecting Passed and ${job1_end_value} observed!" > >(tee -a "${error_log}")
    exit 1
  fi



  kill_agents
  rm -rf /tmp/COMPSsWorker01/*
  AGENT_PIDS=""

  # Copy LOG files
  # cp -rf "${log_dir}" "${target_log_folder}"
  port_offset=$((port_offset + 100 ));
done
exit 0
