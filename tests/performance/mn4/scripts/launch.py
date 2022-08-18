import datetime
import os
import subprocess
from subprocess import PIPE

base_dir = "/gpfs/projects/bsc19/PERFORMANCE/dislib"
tests_dir = os.path.join(base_dir, "tests")
logs_dir = os.path.join(base_dir, "logs")
scripts_dir = os.path.join(base_dir, "scripts")
exec_time = 60


def main():
    cmd = ("enqueue_compss"
           " --exec_time=" + str(exec_time) +
           " --pythonpath=" + scripts_dir + ":" + tests_dir +
           " --lang=python"
           " --worker_in_master_cpus=0"
           " --max_tasks_per_node=48"
           " --num_nodes=9 ").split(" ")

    out = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    logdir = os.path.join(logs_dir, out)
    os.makedirs(logdir, exist_ok=True)
    cmd.append("--master_working_dir=" + logdir)

    gpfs_cmd = cmd + ["--worker_working_dir=" + logs_dir]
    scratch_cmd = cmd + ["--worker_working_dir=local_disk"]
    dependencies = "afterany"

    for f in os.listdir(tests_dir):
        if f.endswith(".py"):
            test_path = os.path.join(tests_dir, f)

            print("Submitting " + f)

            job_id = run_job(gpfs_cmd + [test_path])
            dependencies = dependencies + ":" + job_id

            job_id = run_job(scratch_cmd + [test_path])
            dependencies = dependencies + ":" + job_id

    final_cmd = ["sbatch", "-n1", "--dependency=" + dependencies,
                 os.path.join(scripts_dir, "postprocess.sh"), out]
    subprocess.run(final_cmd)


def run_job(cmd):
    proc = subprocess.run(args=cmd, stdout=PIPE)
    job_id = str(proc.stdout).split(" ")[-1][:-3]
    return job_id


if __name__ == "__main__":
    main()
