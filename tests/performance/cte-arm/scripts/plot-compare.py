import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("pdf")


def main():
    """
    Plots two different sets of executions.

    Arguments
    ---------
    name1 : fist execution name to display in plot (e.g., dislib-v0.3)
    timestamp1 : first execution timestamp (e.g., 2019-08-12_12:07)
    name2: second execution name to display in plot
    timestamp2 : second execution timestamp
    """
    import matplotlib.pyplot as plt

    name1 = sys.argv[1]
    ts1 = sys.argv[2]
    name2 = sys.argv[3]
    ts2 = sys.argv[4]

    base_dir = "/fefs/scratch/bsc19/bsc19029/PERFORMANCE/dislib"
    plot_path = os.path.join(base_dir, "plots")
    res_path = os.path.join(base_dir, "results")

    os.makedirs(plot_path, exist_ok=True)
    out = os.path.join(plot_path, name1 + "-vs-" + name2)

    df1_g, df1_s, names1 = read_results(os.path.join(res_path, ts1))
    df2_g, df2_s, names2 = read_results(os.path.join(res_path, ts2))

    names = pd.concat([names1, names2]).drop_duplicates(
        subset="name").reset_index()

    gen_plot(df1_g, df2_g, name1, name2, names, plt)
    plt.savefig(out + "-gpfs.pdf", dpi=1200, pad_inches=0.1,
                bbox_inches="tight")

    plt.cla()

    gen_plot(df1_s, df2_s, name1, name2, names, plt)
    plt.savefig(out + "-scratch.pdf", dpi=1200, pad_inches=0.1,
                bbox_inches="tight")


def gen_plot(df1, df2, name1, name2, names, plt):
    width = 0.35
    n = names.shape[0]
    times1 = np.zeros(n)
    times2 = np.zeros(n)
    for index, row in names.iterrows():
        name = row["name"]

        if df1[df1["name"] == name]["time"].size > 0:
            times1[index] = df1[df1["name"] == name]["time"].iloc[0]

        if df2[df2["name"] == name]["time"].size > 0:
            times2[index] = df2[df2["name"] == name]["time"].iloc[0]
    ind = np.arange(names.shape[0])
    plt.bar(ind - width / 2, times1, width=width, label=name1)
    plt.bar(ind + width / 2, times2, width=width, label=name2)
    xlabels = names["name"].values
    plt.xticks(ind, xlabels, rotation=45, ha="right")
    plt.xlabel("Algorithm")
    plt.ylabel("Time (s)")
    plt.legend(loc="best", ncol=2)


def read_results(path):
    df = pd.read_csv(path, sep=" ", names=["algorithm", "storage",
                                           "dataset", "time"])
    df = df[df.dataset != "ERROR"]
    df = df.sort_values(by="algorithm", axis=0)
    df["name"] = df.apply(lambda row: str(row["algorithm"] + "-" + row[
        "dataset"]), axis=1)

    names = df["name"].drop_duplicates().reset_index()

    df_g = df[df.storage == "gpfs"]
    df_s = df[df.storage == "scratch"]

    return df_g, df_s, names


if __name__ == "__main__":
    main()
