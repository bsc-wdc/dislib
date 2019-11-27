import os
import sys


def main():
    res_path = "/gpfs/projects/bsc19/PERFORMANCE/dislib/results"
    log_path = "/gpfs/projects/bsc19/PERFORMANCE/dislib/logs"
    gpfs_path = "/gpfs/scratch/bsc19/compss/COMPSs_Sandbox"

    os.makedirs(res_path, exist_ok=True)
    out = os.path.join(res_path, sys.argv[1])
    log_dir = os.path.join(log_path, sys.argv[1])

    for f in os.listdir(log_dir):
        if f.endswith(".out"):
            wd = "scratch"

            for line in open(os.path.join(log_dir, f), "r"):
                if gpfs_path in line:
                    wd = "gpfs"
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
