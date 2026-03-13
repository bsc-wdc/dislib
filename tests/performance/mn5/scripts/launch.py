import datetime
import os
import subprocess
from subprocess import PIPE

base_dir = "/gpfs/projects/bsc19/PERFORMANCE/dislib/dislib"

try:
    import dislib
    print("Using already loaded dislib:", dislib.__version__)
    dislib_lib_path = ""
except ImportError:
    dislib_lib_path = ":" + base_dir

tests_dir = os.path.join(base_dir, "tests/performance/mn5/tests")
logs_dir = os.path.join(base_dir, "logs")
scripts_dir = os.path.join(base_dir, "tests/performance/mn5/scripts")
preimports_path = os.path.join(scripts_dir, "preimports.txt")
exec_time = 120
scheduler = "es.bsc.compss.scheduler.orderstrict.fifo.FifoTS"


def main():
    cmd = ("enqueue_compss"
           " --exec_time=" + str(exec_time) +
           " --scheduler=" + scheduler +
           " --project_name=bsc19" +
           " --qos=gp_debug" +
           " --tracing" +
           " --pythonpath=" + scripts_dir + ":" + tests_dir + dislib_lib_path +
           " --lang=python"
           " --worker_in_master_cpus=48"
           " --max_tasks_per_node=112"
           " --num_nodes=4 ").split(" ")

    out = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    logdir = os.path.join(logs_dir, out)
    os.makedirs(logdir, exist_ok=True)
    cmd.append("--job_execution_dir=" + logdir)
    cmd.append("--base_log_dir=" + logs_dir)

    gpfs_cmd = cmd + [
        "--worker_working_dir=" + logs_dir]
    scratch_cmd = cmd + ["--worker_working_dir=local_disk"]
    dependencies = "afterany"
    job_id = None
    for f in os.listdir(tests_dir):
        if f.endswith(".py"):
            test_path = os.path.join(tests_dir, f)

            print("Submitting " + f)

            if job_id is not None:
                gpfs_dep_cmd = gpfs_cmd + ["--job_dependency="+job_id]
                job_id = run_job(gpfs_dep_cmd + [test_path])
                dependencies = dependencies + ":" + job_id
                scratch_dep_cmd = scratch_cmd + ["--job_dependency="+job_id]
                job_id = run_job(scratch_dep_cmd + [test_path])
            else:
                job_id = run_job(gpfs_cmd + [test_path])
                dependencies = dependencies + ":" + job_id
                scratch_dep_cmd = scratch_cmd + ["--job_dependency="+job_id]
                job_id = run_job(scratch_dep_cmd + [test_path])
                dependencies = dependencies + ":" + job_id

    final_cmd = ["sbatch", "-n1", "--dependency=" + dependencies,
                 os.path.join(scripts_dir, "postprocess.sh"), out]
    subprocess.run(final_cmd)


def run_job(cmd):
    os.environ['PRELOAD_PYTHON_LIBRARIES'] = preimports_path
    print("Running command:\n", " ".join(cmd))
    proc = subprocess.run(args=cmd, stdout=PIPE, env=os.environ.copy())
    job_id = str(proc.stdout).split(" ")[-1][:-3]
    return job_id


if __name__ == "__main__":
    main()
