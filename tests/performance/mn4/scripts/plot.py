import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("pdf")


def main():
    import matplotlib.pyplot as plt

    plot_path = "/gpfs/projects/bsc19/PERFORMANCE/dislib/plots"
    res_path = "/gpfs/projects/bsc19/PERFORMANCE/dislib/results"

    os.makedirs(plot_path, exist_ok=True)
    out = os.path.join(plot_path, sys.argv[1])
    results = os.path.join(res_path, sys.argv[1])

    df = pd.read_csv(results, sep=" ", names=["algorithm", "storage",
                                              "dataset", "time"])
    df = df[df.dataset != "ERROR"]
    df = df.sort_values(by="algorithm", axis=0)
    df["name"] = df.apply(lambda row: str(row["algorithm"] + "-" + row[
        "dataset"]), axis=1)

    names = df["name"].drop_duplicates().reset_index()

    df_g = df[df.storage == "gpfs"]
    df_s = df[df.storage == "scratch"]

    width = 0.35

    n = names.shape[0]

    times_g = np.zeros(n)
    times_s = np.zeros(n)

    for index, row in names.iterrows():
        name = row["name"]

        if df_g[df_g["name"] == name]["time"].size > 0:
            times_g[index] = df_g[df_g["name"] == name]["time"].iloc[0]

        if df_s[df_s["name"] == name]["time"].size > 0:
            times_s[index] = df_s[df_s["name"] == name]["time"].iloc[0]

    ind = np.arange(names.shape[0])

    plt.bar(ind - width / 2, times_g, width=width, label="gpfs")
    plt.bar(ind + width / 2, times_s, width=width, label="scratch")

    xlabels = names["name"].values

    plt.xticks(ind, xlabels, rotation=45, ha="right")
    plt.xlabel("Algorithm")
    plt.ylabel("Time (s)")

    plt.legend(loc="best", ncol=2)

    plt.savefig(out + ".pdf", dpi=1200, pad_inches=0.1, bbox_inches="tight")


if __name__ == "__main__":
    main()
