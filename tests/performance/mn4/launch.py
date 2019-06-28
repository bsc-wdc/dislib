import datetime
import os
import subprocess
from subprocess import PIPE

tests_dir = "/gpfs/projects/bsc19/PERFORMANCE/dislib/tests"


def main():
    cmd = "enqueue_compss --exec_time=60 " \
          "--pythonpath=/gpfs/projects/bsc19/PERFORMANCE/dislib/scripts" \
          ":/gpfs/projects/bsc19/PERFORMANCE/dislib/tests --lang=python " \
          "--worker_in_master_cpus=0 --max_tasks_per_node=48 --num_nodes=9 " \
          "--master_working_dir=/gpfs/projects/bsc19/PERFORMANCE/dislib/logs" \
          "".split(" ")

    out = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    logdir = os.path.join("/gpfs/projects/bsc19/PERFORMANCE/dislib/logs/", out)
    os.makedirs(logdir, exist_ok=True)
    cmd.append("--master_working_dir=" + logdir)

    gpfs_cmd = cmd + [
        "--worker_working_dir=/gpfs/scratch/bsc19/compss/COMPSs_Sandbox"]
    scratch_cmd = cmd + ["--worker_working_dir=scratch"]
    dependencies = "afterany"

    for f in os.listdir(tests_dir):
        if f.endswith(".py"):
            script_path = os.path.join(tests_dir, f)

            print("Submitting " + f)

            job_id = run_job(gpfs_cmd + [script_path])
            dependencies = dependencies + ":" + job_id

            job_id = run_job(scratch_cmd + [script_path])
            dependencies = dependencies + ":" + job_id

    final_cmd = ["sbatch", "-n1", "--dependency=" + dependencies,
                 "/gpfs/projects/bsc19/PERFORMANCE/dislib/scripts"
                 "/postprocess.sh",
                 out]
    subprocess.run(final_cmd)


def run_job(cmd):
    proc = subprocess.run(args=cmd, stdout=PIPE)
    job_id = str(proc.stdout).split(" ")[-1][:-3]
    return job_id


if __name__ == "__main__":
    main()
