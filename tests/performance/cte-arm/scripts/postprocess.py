import os
import sys


def main():
    base_dir = "/fefs/scratch/bsc19/bsc19029/PERFORMANCE/dislib"
    res_path = os.path.join(base_dir, "results")
    log_path = os.path.join(base_dir, "logs")
    shared_path = os.path.join(base_dir, "logs")

    os.makedirs(res_path, exist_ok=True)
    out = os.path.join(res_path, sys.argv[1])
    log_dir = os.path.join(log_path, sys.argv[1])

    for f in os.listdir(log_dir):
        if f.endswith(".out"):
            wd = "tmp"

            for line in open(os.path.join(log_dir, f), "r"):
                if shared_path in line:
                    wd = "fefs"
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
