#!/bin/bash
# coses script:
#     parametritzar les coses amb flags -- done
#     cpus dels agents no hardcoded -- done
#     comprobar si topologia s'ha creat bé -- done
#     que cada execució et surti en una carpeta diferenta -- done
#     identificar si ha acabat amb el programa bé i han mort els agents sols o si han tinugt que matar als agents -- done
#     tornar a afegir project i resoruces -- done
# Traps and Handlers
usage() {
    echo ": number of agents"
    echo ": define the topology of agents. plain (default), tree, chain"
    echo ": Llenguatge"
    echo ": Path"
    echo ": Executable"
    echo ": executions params"
    exit 1
}

kill_agents() {
    echo ""
    echo "---------------------------------------------------------------------------"
    echo "----------------------------- KILLING AGENTS ----------------------------"
    echo "---------------------------------------------------------------------------"
    echo ""
    compss_clean_procs
}
trap kill_agents SIGINT

get_args() {
    while getopts hvgtmd-: flag; do
        case "$flag" in
            h)
                # TODO usage
                ;;
            -)
                case "$OPTARG" in
                    cei=*)
                        cei=${OPTARG//cei=/}
                        ;;
                    stop)
                        echo _______ asignant acction=true
                        action="true"
                        ;;
                    lang=*)
                        lang=${OPTARG//lang=/}
                        ;;
                    method_name=*)
                        method_name=${OPTARG//method_name=/}
                        ;;
                    array)
                        array_param="true"
                        ;;
                    parameters_array)
                        array_param="true"
                        ;;
                    num_agents=*)
                        num_agents=${OPTARG//num_agents=/}
                        ;;
                    topology=*)
                        topology=${OPTARG//topology=/}
                        ;;
                    project=*)
                        project=${OPTARG//project=/}
                        args_start="$args_start --$OPTARG"
                        ;;
                    *)
                        args_start="$args_start --$OPTARG"
                        ;;
                esac
                ;;
            *)
                args_start="$args_start -$flag"
                ;;
        esac
    done
    shift $((OPTIND-1))
    executable=$1
    app_params=${OPTIND}
}
call_operation() {
    echo ""
    echo "---------------------------------------------------------------------------"
    echo "----------------------------- CALLING OPERATION ----------------------------"
    echo "---------------------------------------------------------------------------"
    echo ""

    echo _______ comprovant acction
    if [ "${action}" == "true" ]; then
        echo _______ comprovant acction es igual = true

        if [ ${num_agents} -gt 1 ]; then
            for ((i=1;i<=max_agent_num;i++)); do
                workerAgents=${workerAgents}"${arr_agents[$i]}:${arr_rest[$i]};"
            done
            workerAgents=${workerAgents::-1} #delete last semicolon
            end_agents="
            --forward_to=$workerAgents"
        fi
        end_agents="--stop $end_agents"
    fi
    if [ ! -z ${method_name} ]; then
        method_name="--method_name=${method_name}"
    fi
    if [ ! -z ${cei} ]; then
        cei="--cei=${cei}"
    fi
    if [ ${array_param} == "true" ]; then
        params_array="--array"
    fi
    callOperationCommand="${COMPSS_HOME}/Runtime/scripts/user/compss_agent_call_operation
        --lang=${lang^^}
        --master_node=127.0.0.1
        --master_port=${arr_rest[0]}
        ${method_name} ${cei} ${end_agents} ${params_array}
        ${executable} ${execution_params}"
        printf '%b\n' "${callOperationCommand} >>${output_log} 2>>${error_log}"
        ${callOperationCommand} >>${output_log} 2>>${error_log}
    retries=${default_retries}
    while [ ! -f ${log_dir}/${arr_agents[0]}/jobs/job1_NEW.out ]; do
        sleep 3
        retries=$((retries - 1 ))
    done
    echo ${log_dir}/${arr_agents[0]}/jobs/job1_NEW.out
    echo ""
    if [ -f ${log_dir}/${arr_agents[0]}/jobs/job1_NEW.out ]; then
        echo "${GREEN}Execution properly started${NC}"
        echo ""
    else
        echo "Call operation didn't start any job"
        exit 1
    fi
    echo "waiting for agent1 with pid: ": $pidAgent1
    echo ""
    wait $pidAgent1
    if grep -q "Job completed after" "${log_dir}/${arr_agents[0]}.outputlog"; then
        echo "${GREEN}Execution ended succesfully${NC}"
    else
        echo "${RED}Execution failed${NC}"
    fi
    echo ""
}
compss_clean_procs
sleep 0.5
# rm -rf ~/.COMPSs
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
NC=$(tput sgr0)
default_retries="10"
default_name="COMPSsWorker"
default_rest="46001"
default_comm="46002"
project="/home/bscuser/git/dislib/tests/performance/mn4/tests/projectAgents4cpu.xml"
log_dir="$HOME/.COMPSs/$(date +"%Y.%m.%d.%H%M%S")"
mkdir -p "${log_dir}"
output_log="${log_dir}/outputlog"
error_log="${log_dir}/errorlog"
touch "${output_log}"
touch "${error_log}"
arr_agents=()
arr_rest=()
arr_comm=()
############################################################################################################
#########################                      NESTED AGENTS                       #########################
############################################################################################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# echo ______ resources path ${SCRIPT_DIR}/resources.xml
# echo ______ resources content ; cat  ${SCRIPT_DIR}/resources.xml
# echo ______ projectAgents1cpu path ${SCRIPT_DIR}/projectAgents1cpu.xml
# echo ______ projectAgents1cpu content ; cat  ${SCRIPT_DIR}/projectAgents1cpu.xml
get_args "$@"
if [ ! -z  ${executable} ]; then
    shift ${app_params}
    execution_params=$@
fi

num_cpu=`grep '<Computing' $project | cut -f2 -d">"|cut -f1 -d"<"`

let "max_agent_num = $num_agents - 1"
# if [ $# -lt 4 ]; then
#     usage
# fi
echo ""
echo "---------------------------------------------------------------------------"
echo "------------------------ STARTING ${num_agents} AGENTS -----------------------"
echo "---------------------------------------------------------------------------"
echo ""
shopt -s nocasematch
for i in $(seq 1 $num_agents)
do
    # define agent name with 2 digits
    if [ $i -lt 10 ]
    then
        agent_name="${default_name}0${i}"
        arr_agents+=(${agent_name})
    else
        agent_name="${default_name}${i}"
        arr_agents+=(${agent_name})
    fi
    log_agent="${log_dir}/${agent_name}"
    # delete previous Agent dir
    if [ -d ${log_agent} ]; then rm -r ${log_agent}; fi
    # port calculation
    let "rest_port = ${default_rest} + $i * 100"
    arr_rest+=(${rest_port})
    let "comm_port = ${default_comm} + $i * 100"
    arr_comm+=(${comm_port})
    agentStartCommand="
    compss_agent_start
        --hostname=${agent_name}
        --log_dir=${log_agent}
        --rest_port=${rest_port}
        --comm_port=${comm_port}
        ${args_start}
    "
    printf '%b\n' "${agentStartCommand} 1>"${log_dir}/${agent_name}.outputlog" 2>"${log_dir}/${agent_name}.errorlog" &"
    ${agentStartCommand} 1>"${log_dir}/${agent_name}.outputlog" 2>"${log_dir}/${agent_name}.errorlog" &
    if [ ! $pidAgent1 ]; then
        pidAgent1=$!

    fi
    echo ""
done
for i in $(seq 0 $max_agent_num)
do
    retries=${default_retries}
    curl -XGET http://127.0.0.1:${arr_rest[$i]}/COMPSs/test 1>/dev/null 2>/dev/null
    while [ ! "$?" == "0" ] && [ "${retries}" -gt "0" ]; do
        sleep 3
        retries=$((retries - 1 ))
        curl -XGET http://127.0.0.1:${arr_rest[$i]}/COMPSs/test 1>/dev/null 2>/dev/null
    done
    sleep 1
    RESULT=$(grep "test invoked" "${log_dir}/${arr_agents[$i]}.outputlog")
    if [ -z "${RESULT}" ]; then
        echo "${arr_agents[$i]} failed to start" | tee -a "${error_log}"
        exit 1
    fi
    echo "${arr_agents[$i]} started" >>"${output_log}"
    echo "${GREEN}${arr_agents[$i]} started${NC}"
    echo ""
done

if [ ${num_agents} -gt 1 ]; then
    echo ""
    echo "---------------------------------------------------------------------------"
    echo "------------------ ADDING AGENTS AS RESOURCES AS ${topology^^} -----------------"
    echo "---------------------------------------------------------------------------"
    echo ""
fi
if [[ $topology = "tree" ]]
then
    agent_cpu=()
    for i in $(seq 1 $num_agents)
    do
        agent_cpu+=($num_cpu)
    done
    for i in $(seq $max_agent_num -1 1)
    do
        # calculte the parent node
        agent=$(echo "($i + 1)/2" | bc)
        # add to the parent node the number of cpus
        let "agent_cpu[${agent}-1] += agent_cpu[$i]"
        compss_agent_add_resources\
            --agent_node=127.0.0.1\
            --agent_port="${arr_rest[${agent}-1]}"\
            --comm="es.bsc.compss.agent.comm.CommAgentAdaptor"\
            --cpu=${agent_cpu[$i]}\
            ${arr_agents[$i]} Port=${arr_comm[$i]}
    done
elif [[ $topology = "chain" ]]
then
    sum_cpu=$num_cpu
    for i in $(seq $max_agent_num -1 1)
    do
        compss_agent_add_resources\
            --agent_node=127.0.0.1\
            --agent_port="${arr_rest[${i}-1]}"\
            --comm="es.bsc.compss.agent.comm.CommAgentAdaptor"\
            --cpu=${sum_cpu}\
            ${arr_agents[$i]} Port=${arr_comm[$i]}
            let "sum_cpu++"
    done
elif [[ $topology = "plain" ]] || [[ -z $topology ]]
then
    for i in $(seq 1 $max_agent_num)
    do
        compss_agent_add_resources\
            --agent_node=127.0.0.1\
            --agent_port="${arr_rest[0]}"\
            --comm="es.bsc.compss.agent.comm.CommAgentAdaptor"\
            --cpu=${num_cpu}\
            ${arr_agents[$i]} Port=${arr_comm[$i]}
    done
else
    echo "Topology not supported."
    echo "Available options: tree, chain, plain (default) "
    exit 1
fi

cpus_topology=`curl -s -XGET http://127.0.0.1:46101/COMPSs/resources | jq | grep "units" | cut -d ":" -f 2 | sed 's/,//' | sed ':a;N;$!ba;s/\n/+/' | bc`
let "cpus_num_agents = $num_agents * $num_cpu"
echo ""
if [[ ${cpus_topology} -eq ${cpus_num_agents} ]]; then
    echo "${GREEN}Topology created successfully${NC}"
else
    echo "${RED}Error creating topology${NC}"
fi
echo ""

sleep 1
if [ ${num_agents} -gt 1 ]; then
    resources=`curl -XGET -s http://127.0.0.1:46101/COMPSs/resources` 2>> /dev/null
fi
if [ ! -z ${executable} ]; then
    call_operation
fi
