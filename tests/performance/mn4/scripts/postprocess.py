import os
import sys


def main():
    res_path = "/gpfs/projects/bsc19/PERFORMANCE/dislib/results"
    log_path = "/gpfs/projects/bsc19/PERFORMANCE/dislib/logs"

    os.makedirs(res_path, exist_ok=True)
    out = os.path.join(res_path, sys.argv[1])
    log_dir = os.path.join(log_path, sys.argv[1])

    for f in os.listdir(log_dir):
        if f.endswith(".out"):
            time = -1

            for line in open(os.path.join(log_dir, f), "r"):
                if "Worker WD:" in line:
                    if "gpfs" in str(line.split(" ")[-1]):
                        wd = "gpfs"
                    else:
                        wd = "scratch"
                elif "==== STARTING ====" in line:
                    time = -1
                    test_name = str(line.split(" ")[-1]).rstrip()
                elif "==== OUTPUT ====" in line:
                    time = float(line.split(" ")[-1])
                    dataset = str(line.split(" ")[-2])

                    with open(out, "a") as fout:
                        fout.write(
                            test_name + " " + wd + " " + dataset + " " + str(
                                time) + "\n")

            if time == -1:
                with open(out, "a") as fout:
                    fout.write(test_name + " " + wd + " ERROR\n")


if __name__ == "__main__":
    main()
