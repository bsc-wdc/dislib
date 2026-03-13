import os
import sys


def main():
    res_path = "/gpfs/projects/bsc19/PERFORMANCE/dislib/dislib/results"
    log_path = "/gpfs/projects/bsc19/PERFORMANCE/dislib/dislib/logs"
    line_to_check = "In worker_working_dir:"

    os.makedirs(res_path, exist_ok=True)
    out = os.path.join(res_path, sys.argv[1])
    log_dir = os.path.join(log_path, sys.argv[1])

    for f in os.listdir(log_dir):
        if f.endswith(".out"):

            for line in open(os.path.join(log_dir, f), "r"):
                if line_to_check in line:
                    if "/gpfs" in line:
                        wd = "gpfs"
                    elif "/scratch/tmp" in line:
                        wd = "scratch"
                    else:
                        raise ValueError(
                            "Invalid worker_working_dir in log file:"
                            " expected path under /gpfs or /scratch/tmp")
                    break

            test_name = ""
            dataset = ""

            for line in open(os.path.join(log_dir, f), "r"):
                if "==== STARTING ====" in line:
                    if test_name:
                        output = (test_name + " "
                                  + wd + " "
                                  + dataset + " ERROR\n")

                        with open(out, "a") as fout:
                            fout.write(output)

                    dataset = str(line.split(" ")[-1]).rstrip()
                    test_name = str(line.split(" ")[-2])
                elif "==== TIME ====" in line:
                    time = float(line.split(" ")[-1])

                    output = (test_name + " "
                              + wd + " "
                              + dataset + " "
                              + str(time) + "\n")

                    with open(out, "a") as fout:
                        fout.write(output)

                    test_name = ""
                    dataset = ""


if __name__ == "__main__":
    main()
