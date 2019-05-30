import os
import sys


def main():
    out = "/gpfs/projects/bsc19/PERFORMANCE/dislib/results/" + sys.argv[1]
    log_dir = os.path.join("/gpfs/projects/bsc19/PERFORMANCE/dislib/logs",
                           sys.argv[1])

    for f in os.listdir(log_dir):
        if f.endswith(".out"):
            time = "ERROR"

            for line in open(os.path.join(log_dir, f), "r"):
                if "==== STARTING ====" in line:
                    test_name = str(line.split(" ")[-1]).rstrip()
                elif "==== OUTPUT ====" in line:
                    time = float(line.split(" ")[-1])
                elif "Worker WD:" in line:
                    if "gpfs" in str(line.split(" ")[-1]):
                        wd = "gpfs"
                    else:
                        wd = "scratch"

            with open(out, "a") as fout:
                fout.write(test_name + " " + wd + " " + str(time) + "\n")


if __name__ == "__main__":
    main()
