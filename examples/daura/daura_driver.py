#!/usr/bin/env python3
import argparse
import numpy as np
import pytraj as pt
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task
from dislib.data.array import Array
from dislib.cluster import Daura


def main(traj_path, top_path, cutoff):
    # Load trajectory with topology.
    # pt.load loads into memory, should be replaced
    traj = pt.load(traj_path, top=top_path)
    n_frames = traj.n_frames
    step = 20  # number of frames per chunk
    dist_blocks = []
    for i in range(0, n_frames, step):
        dist_row = []
        for j in range(0, n_frames, step):
            dist_row.append(compute_rmsd(i, j, step, traj_path, top_path))
        dist_blocks.append(dist_row)
    reg_shape = (step, step)
    shape = (n_frames, n_frames)
    distances = Array(dist_blocks, reg_shape, reg_shape, shape, False)
    daura = Daura(cutoff=cutoff)
    clusters = daura.fit_predict(distances)
    clusters = compss_wait_on(clusters)
    print('Cluster sizes: ', [len(c) for c in clusters])
    print('Clusters (first frame is the center):', clusters)


@task(traj_path=FILE_IN, top_path=FILE_IN, returns=1)
def compute_rmsd(i, j, step, traj_path, top_path):
    # Load trajectory with topology.
    # pt.load loads into memory, should be replaced
    traj = pt.load(traj_path, top=top_path)
    t1 = traj[i:i + step]
    t2 = traj[j:j + step]
    block_rmsd = np.empty(shape=(len(t1), len(t2)))
    for i in range(len(t1)):
        # For some reason pytraj results are off by a 10 factor
        block_rmsd[i] = pt.rmsd(traj=t2, ref=t1[i], mass=True) / 10
    return block_rmsd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="Trajectory: .xtc")
    parser.add_argument("-s", type=str, help="Structure: .pdb")
    parser.add_argument("-cutoff", "--cutoff", metavar="CUTOFF", type=float,
                        help="RMSD cut-off (nm) for two structures to be "
                             "neighbors")
    args = parser.parse_args()
    main(args.f, args.s, args.cutoff)
