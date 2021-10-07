#!/usr/bin/env python3
import argparse
import numpy as np
import pytraj as pt
import os

from pycompss.api.constraint import constraint
from pytraj.utils.tools import split_and_write_traj

from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import FILE_IN
from pycompss.api.task import task

from dislib.cluster import Daura
from dislib.data.array import Array


def main(traj_path, top_path, n_chunks, cutoff):
    # Load trajectory with topology. Using iterload for memory saving.
    traj = pt.iterload(traj_path, top=top_path)
    n_frames = traj.n_frames
    chunksize = n_frames // n_chunks
    chunks_dir = 'traj_chunks'
    os.makedirs(chunks_dir, exist_ok=True)
    for f_name in os.listdir(chunks_dir):
        if f_name[:5] == 'trajx':
            os.remove('/'.join((chunks_dir, f_name)))
    split_and_write_traj(traj, n_chunks=n_chunks,
                         root_name='/'.join(('.', chunks_dir, 'trajx')),
                         ext='xtc')
    files = os.listdir(chunks_dir)
    chunk_files = []
    i = 0
    while True:
        f_name = '.'.join(('trajx', str(i), 'xtc'))
        if f_name in files:
            chunk_files.append(f_name)
            i += 1
        else:
            break
    dist_blocks = []
    for chunk1 in chunk_files:
        chunk1 = '/'.join((chunks_dir, chunk1))
        dist_row = []
        for chunk2 in chunk_files:
            chunk2 = '/'.join((chunks_dir, chunk2))
            dist_row.append(compute_rmsd(chunk1, chunk2, top_path))
        dist_blocks.append(dist_row)
    shape = (n_frames, n_frames)
    reg_shape = (chunksize, chunksize)
    distances = Array(dist_blocks, reg_shape, reg_shape, shape, False)
    daura = Daura(cutoff=cutoff)
    clusters = daura.fit_predict(distances)
    clusters = compss_wait_on(clusters)
    print('Cluster sizes: ', [len(c) for c in clusters])
    print('Clusters (first frame is the center):', clusters)


@constraint(computing_units="${ComputingUnits}")
@task(chunk1=FILE_IN, chunk2=FILE_IN, top_path=FILE_IN, returns=1)
def compute_rmsd(chunk1, chunk2, top_path):
    # Load trajectory chunks with topology.
    t1 = pt.load(chunk1, top=top_path)
    t2 = pt.load(chunk2, top=top_path)
    block_rmsd = np.empty(shape=(len(t1), len(t2)))
    for i in range(len(t1)):
        # Compute the distances and rescale by a factor of 10
        block_rmsd[i] = pt.rmsd(traj=t2, ref=t1[i], mass=True) / 10
    return block_rmsd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", metavar="TRAJECTORY_FILE", type=str,
                        help="Trajectory: .xtc")
    parser.add_argument("-s",  metavar="TOPOLOGY_FILE", type=str,
                        help="Structure: .pdb")
    parser.add_argument("-n", "--n_chunks", metavar="N_CHUNKS", type=int,
                        help='Number of chunks to split the trajectory. '
                             'This determines the tasks granularity.'
                             'The number of chunks may be higher if the split '
                             'is not exact.')
    parser.add_argument("-cutoff", "--cutoff", metavar="CUTOFF", type=float,
                        help="RMSD cut-off (nm) for two structures to be "
                             "neighbors")
    args = parser.parse_args()
    main(args.f, args.s, args.n_chunks, args.cutoff)
