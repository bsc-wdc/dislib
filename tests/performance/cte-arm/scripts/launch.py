import datetime
import os
import subprocess

source_python = 'source /fefs/scratch/bsc19/bsc19776/opt/' \
                'virtual_python-3.6.8/bin/activate'
base_dir = "/fefs/scratch/bsc19/bsc19029/PERFORMANCE/dislib"
tests_dir = os.path.join(base_dir, "tests")
logs_dir = os.path.join(base_dir, "logs")
scripts_dir = os.path.join(base_dir, "scripts")
exec_time = 60
# exec_time = 720
scheduler = "es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler"


def main():
    cmd = ("LC_ALL=en_US.UTF-8 enqueue_compss"
           " --exec_time=" + str(exec_time) +
           " --scheduler=" + scheduler +
           " --pythonpath=" + scripts_dir + ":" + tests_dir + ":" +
           "/fefs/scratch/bsc19/bsc19776/opt/dislib" +
           " --job_dependency=\"performance\""
           " --python_interpreter=python3"
           " --cluster=rscunit_ft02"
           " --queue=def_grp"
           " --sc_cfg=cte-arm.cfg"
           " --cpu_affinity=\"12-59\""
           " --extrae_config_file=/fefs/scratch/bsc19/bsc19776/experiments/"
           "test/extrae_basic.xml"
           " --worker_in_master_cpus=0"
           " --max_tasks_per_node=24"
           " --num_nodes=17 ").split(" ")

    out = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    logdir = os.path.join(logs_dir, out)
    os.makedirs(logdir)
    cmd.append("--master_working_dir=" + logdir)
    cmd.append("--base_log_dir=" + logdir)

    gpfs_cmd = cmd + [
        "--worker_working_dir=" + logdir]
    # scratch_cmd = cmd + ["--worker_working_dir=/tmp"]

    for f in os.listdir(tests_dir):
        if f.endswith(".py"):
            test_path = os.path.join(tests_dir, f)

            print("Submitting " + f)

            run_job(gpfs_cmd + [test_path], logdir)

            # Do not run in /tmp
            # run_job(scratch_cmd + [test_path], logdir)

    final_cmd = ["pjsub", "-L", "node=1", "--step", "--sparam",
                 "jnam=performance", "-x", "time_str=" + out,
                 os.path.join(scripts_dir, "postprocess.sh")]
    # print(' '.join(final_cmd))
    subprocess.run('cd ' + logdir + '; ' + ' '.join(final_cmd), shell=True,
                   executable='/bin/bash')


def run_job(cmd, logdir):
    cd_log = 'cd ' + logdir
    subprocess.run(cd_log + '; ' + source_python + '; ' + ' '.join(cmd) +
                   '; deactivate', shell=True, executable='/bin/bash')


if __name__ == "__main__":
    main()
