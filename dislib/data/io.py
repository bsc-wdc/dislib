import numpy as np
from numpy.lib import format
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Depth, COLLECTION_INOUT, COLLECTION_OUT, \
    Type, FILE_IN, IN_DELETE, COLLECTION_FILE_IN
from pycompss.api.task import task
import os

from dislib.data.array import Array
from math import ceil
_CRD_LINE_SIZE = 81


def load_svmlight_file(path, block_size, n_features, store_sparse):
    """ Loads a SVMLight file into a distributed array.

    Parameters
    ----------
    path : string
        File path.
    block_size : tuple (int, int)
        Size of the blocks for the output ds-array.
    n_features : int
        Number of features.
    store_sparse : boolean
        Whether to use scipy.sparse data structures to store data. If False,
        numpy.array is used instead.

    Returns
    -------
    x, y : (ds-array, ds-array)
        A distributed representation (ds-array) of the X and y.
    """
    n, m = block_size
    lines = []
    x_blocks, y_blocks = [], []

    n_rows = 0
    with open(path, "r") as f:
        for line in f:
            n_rows += 1
            lines.append(line.encode())

            if len(lines) == n:
                # line 0 -> X, line 1 -> y
                out_blocks = Array._get_out_blocks((1, ceil(n_features / m)))
                out_blocks.append([object()])
                # out_blocks.append([])
                _read_svmlight(lines, out_blocks, col_size=m,
                               n_features=n_features,
                               store_sparse=store_sparse)
                # we append only the list forming the row (out_blocks depth=2)
                x_blocks.append(out_blocks[0])
                y_blocks.append(out_blocks[1])
                lines = []

    if lines:
        out_blocks = Array._get_out_blocks((1, ceil(n_features / m)))
        out_blocks.append([object()])
        _read_svmlight(lines, out_blocks, col_size=m,
                       n_features=n_features, store_sparse=store_sparse)
        # we append only the list forming the row (out_blocks depth=2)
        x_blocks.append(out_blocks[0])
        y_blocks.append(out_blocks[1])

    x = Array(x_blocks, top_left_shape=block_size, reg_shape=block_size,
              shape=(n_rows, n_features), sparse=store_sparse)

    # y has only a single line but it's treated as a 'column'
    y = Array(y_blocks, top_left_shape=(n, 1), reg_shape=(n, 1),
              shape=(n_rows, 1), sparse=False)

    return x, y


def load_txt_file(path, block_size, discard_first_row=False,
                  col_of_index=False, delimiter=","):
    """ Loads a text file into a distributed array.

    Parameters
    ----------
    path : string
        File path.
    block_size : tuple (int, int)
        Size of the blocks of the array.
    discard_first_row : bool
        Boolean that indicates if the first row should be discarded.
    col_of_index : bool
        Boolean that indicates if the first column is a column
        of indexes and therefore it should be discarded.
    delimiter : string, optional (default=",")
        String that separates columns in the file.

    Returns
    -------
    x : ds-array
        A distributed representation of the data divided in blocks.
    """

    with open(path, "r") as f:
        first_line = f.readline().strip()
        n_cols = len(first_line.split(delimiter))

    n_blocks = ceil(n_cols / block_size[1])
    blocks = []
    lines = []
    n_lines = 0

    with open(path, "r") as f:
        if discard_first_row:
            f.readline()
        for line in f:
            n_lines += 1
            lines.append(line.encode())

            if len(lines) == block_size[0]:
                out_blocks = [object() for _ in range(n_blocks)]
                _read_lines(lines, block_size[1], delimiter, out_blocks,
                            col_of_index=col_of_index)
                blocks.append(out_blocks)
                lines = []

    if lines:
        out_blocks = [object() for _ in range(n_blocks)]
        _read_lines(lines, block_size[1], delimiter, out_blocks,
                    col_of_index=col_of_index)
        blocks.append(out_blocks)

    if col_of_index:
        n_cols = n_cols - 1

    return Array(blocks, top_left_shape=block_size, reg_shape=block_size,
                 shape=(n_lines, n_cols), sparse=False)


def load_npy_file(path, block_size):
    """ Loads a file in npy format (must be 2-dimensional).

    Parameters
    ----------
    path : str
        Path to the npy file.
    block_size : tuple (int, int)
        Block size of the resulting ds-array.

    Returns
    -------
    x : ds-array
    """
    try:
        fid = open(path, "rb")
        version = format.read_magic(fid)
        format._check_version(version)
        shape, fortran_order, dtype = format._read_array_header(fid, version)

        if fortran_order:
            raise ValueError("Fortran order not supported for npy files")

        if len(shape) != 2:
            raise ValueError("Array is not 2-dimensional")

        if block_size[0] > shape[0] or block_size[1] > shape[1]:
            raise ValueError("Block size is larger than the array")

        blocks = []
        n_blocks = int(ceil(shape[1] / block_size[1]))

        for i in range(0, shape[0], block_size[0]):
            read_count = min(block_size[0], shape[0] - i)
            read_size = int(read_count * shape[1] * dtype.itemsize)
            data = fid.read(read_size)
            out_blocks = [object() for _ in range(n_blocks)]
            _read_from_buffer(data, dtype, shape[1], block_size[1], out_blocks)
            blocks.append(out_blocks)

        return Array(blocks=blocks, top_left_shape=block_size,
                     reg_shape=block_size, shape=shape, sparse=False)
    finally:
        fid.close()


def load_mdcrd_file(path, block_size, n_atoms, copy=False):
    """ Loads a mdcrd trajectory file into a distributed array.

    Parameters
    ----------
    path : string
        File path.
    block_size : tuple (int, int)
        Size of the blocks of the array.
    n_atoms : int
        Number of atoms in the trajectory. Each frame in the mdcrd file has
        3*n_atoms float values (corresponding to 3-dimensional position).
    copy : boolean, default=False
        Send the file to every task, as opposed to reading it once in the
        master program.

    Returns
    -------
    x : ds-array
        A distributed representation of the data divided in blocks.
    """
    n_coord = 3
    line_length = 10

    bytes_per_value = 6
    bytes_per_gap = 2

    n_cols = n_atoms * n_coord
    n_hblocks = ceil(n_cols / block_size[1])
    lines_per_snap = ceil((n_atoms * n_coord) / line_length)

    last_line_length = n_cols % 10
    last_line_size = last_line_length * bytes_per_value + \
        (last_line_length - 1) * bytes_per_gap + 3

    bytes_per_snap = (lines_per_snap - 1) * _CRD_LINE_SIZE + last_line_size
    bytes_per_block = block_size[0] * bytes_per_snap

    if not copy:
        return _load_mdcrd(path, block_size, n_cols, n_hblocks,
                           bytes_per_snap, bytes_per_block)
    else:
        return _load_mdcrd_copy(path, block_size, n_cols, n_hblocks,
                                bytes_per_snap, bytes_per_block)


def load_hstack_npy_files(path, cols_per_block=None):
    """ Loads the .npy files in a directory into a ds-array, stacking them
    horizontally, like (A|B|C). The order of concatenation is alphanumeric.

    At least 1 valid .npy file must exist in the directory, and every .npy file
    must contain a valid array. Every array must have the same dtype, order,
    and number of rows.

    The blocks of the returned ds-array will have the same number of rows as
    the input arrays, and cols_per_block columns, which defaults to the number
    of columns of the first array.

    Parameters
    ----------
    path : string
        Folder path.
    cols_per_block : tuple (int, int)
        Number of columns of the blocks for the output ds-array. If None, the
        number of columns of the first array is used.

    Returns
    -------
    x : ds-array
        A distributed representation (ds-array) of the stacked arrays.
    """
    dirlist = os.listdir(path)
    folder_paths = [os.path.join(path, name) for name in sorted(dirlist)]
    # Full path of .npy files in the folder
    files = [pth for pth in folder_paths
             if os.path.isfile(pth) and pth[-4:] == '.npy']
    # Read the header of the first file to get shape, order, and dtype
    with open(files[0], "rb") as fid:
        version = format.read_magic(fid)
        format._check_version(version)
        shape0, order0, dtype0 = format._read_array_header(fid, version)
    rows = shape0[0]
    if cols_per_block is None:
        cols_per_block = shape0[1]
    # Check that all files have the same number of rows, order and datatype,
    # and store the number of columns for each file.
    files_cols = [shape0[1]]
    for filename in files[1:]:
        with open(filename, "rb") as fid:
            version = format.read_magic(fid)
            format._check_version(version)
            shape, order, dtype = format._read_array_header(fid, version)
            if shape[0] != shape0[0] or order0 != order or dtype0 != dtype:
                raise AssertionError()
            files_cols.append(shape[1])

    # Compute the parameters block_files, start_col and end_col for each block,
    # and call the task _load_hstack_npy_block() to generate each block.
    blocks = []
    file_idx = 0
    start_col = 0
    while file_idx < len(files):
        block_files = [files[file_idx]]
        cols = files_cols[file_idx] - start_col
        while cols < cols_per_block:  # while block not completed
            if file_idx + 1 == len(files):  # last file
                break
            file_idx += 1
            block_files.append(files[file_idx])
            cols += files_cols[file_idx]
        # Compute end_col of last file in block (last block may be smaller)
        end_col = files_cols[file_idx] - max(0, (cols - cols_per_block))
        blocks.append(_load_hstack_npy_block(block_files, start_col, end_col))
        if end_col == files_cols[file_idx]:  # file completed
            file_idx += 1
            start_col = 0
        else:  # file uncompleted
            start_col = end_col

    return Array(blocks=[blocks], top_left_shape=(rows, cols_per_block),
                 reg_shape=(rows, cols_per_block),
                 shape=(rows, sum(files_cols)),
                 sparse=False)


def save_txt(arr, dir, merge_rows=False):
    """
    Save a ds-array by blocks to a directory in txt format.

    Parameters
    ----------
    arr : ds-array
        Array data to be saved.
    dir : str
        Directory into which the data is saved.
    merge_rows : boolean, default=False
        Merge blocks along rows before saving.
    """
    os.makedirs(dir, exist_ok=True)
    if merge_rows:
        for i, h_block in enumerate(arr._iterator(0)):
            path = os.path.join(dir, str(i))
            np.savetxt(path, h_block.collect())
    else:
        for i, blocks_row in enumerate(arr._blocks):
            for j, block in enumerate(blocks_row):
                fname = '{}_{}'.format(i, j)
                path = os.path.join(dir, fname)
                block = compss_wait_on(block)
                np.savetxt(path, block)


def save_npy_file(arr, directory, merge_rows=False):
    """
    Save a ds-array by blocks to a directory in npy format.
    Parameters
    ----------
    arr : ds-array
        Array data to be saved.
    dir : str
        Directory into which the data is saved.
    merge_rows : boolean, default=False
        Merge blocks along rows before saving.
    """
    os.makedirs(directory, exist_ok=True)
    if merge_rows:
        for i, h_block in enumerate(arr._iterator(0)):
            path = os.path.join(directory, str(i))
            np.save(path, h_block.collect())
    else:
        for i, blocks_row in enumerate(arr._blocks):
            for j, block in enumerate(blocks_row):
                fname = '{}_{}'.format(i, j)
                path = os.path.join(directory, fname)
                block = compss_wait_on(block)
                np.save(path, block)


def load_npy_files(path, shape=None):
    """ Loads the .npy files in a directory into a ds-array, stacking them
       in a way that the returned ds-array has the same shape as the one
       specified on array_shape. The order of concatenation is alphanumeric.
       At least 1 valid .npy file must exist in the directory, and every .npy
       file must contain a valid array. Every array must have the same dtype,
       order, and number of rows.
       The blocks of the returned ds-array will have the same number of rows
       as the input arrays, and cols_per_block columns, which defaults to the
       number of columns of the first array.
       Parameters
       ----------
       path : string
           Folder path.
       shape : tuple (int, int)
           Number of rows and columns that the returned ds-array will have.
       Returns
       -------
       x : ds-array
           A distributed representation (ds-array) of the stacked arrays.
       """
    if shape is None:
        raise ValueError("The shape of the final dsarray must be specified")
    dirlist = os.listdir(path)
    folder_paths = [os.path.join(path, name) for name in sorted(dirlist)]
    files = [pth for pth in folder_paths
             if os.path.isfile(pth) and pth[-4:] == '.npy']
    with open(files[0], "rb") as fid:
        version = format.read_magic(fid)
        format._check_version(version)
        shape0, order0, dtype0 = format._read_array_header(fid, version)
    blocks = []
    n_blocks0 = int(ceil(shape[0] / shape0[0]))
    n_blocks1 = int(ceil(shape[1] / shape0[1]))
    for i in range(n_blocks0):
        blocks.append([])
        for j in range(n_blocks1):
            fname = '{}_{}.npy'.format(i, j)
            file_to_load = os.path.join(path, fname)
            blocks[-1].append(np.load(file_to_load))
    return Array(blocks=blocks, top_left_shape=shape0,
                 reg_shape=shape0, shape=shape, sparse=False)


def load_blocks_rechunk(blocks, shape, block_size, new_block_size):
    """ Loads the blocks contained in the parameter blocks into
    an ds-array with reg_shape equal to the block_size specified.
    The blocks are loaded respecting the specified shape for the array.
    Finally a rechunk is performed on the ds-array in order to return
    a ds-array with the block size specified in the parameter new_block_size
           Parameters
           ----------
           blocks : list()
               List of the blocks to be set on the ds-array (They should be
               Future objects).
           shape : tuple (int, int)
               Number of rows and columns that the returned ds-array will have.
           block_size : tuple (int, int)
               Number of rows and columns that each block will contain.
           new_block_size : tuple (int, int)
               Number of rows and columns that will contain each block after
               the rechunk operation.
           Returns
           -------
           x : ds-array
               A distributed representation (ds-array) build with the list
               of blocks, with the corresponding shape and with a reg_shape
               set to new_block_size.
           """

    if shape[0] < new_block_size[0] or shape[1] < new_block_size[1]:
        raise ValueError("The block size requested for rechunk"
                         "is greater than the ds-array")
    number_rows = int(shape[0] / block_size[0])
    number_cols = int(shape[1] / block_size[1])
    final_blocks = [[] for _ in range(number_rows)]
    actual_col = 0
    for i in range(number_rows):
        for col in range(number_cols):
            final_blocks[i].append(blocks[actual_col])
            actual_col = actual_col + 1
    arr = _load_blocks_array(final_blocks, shape, block_size)
    return arr.rechunk(new_block_size)


@constraint(computing_units="${ComputingUnits}")
@task(out_blocks=COLLECTION_OUT)
def _read_from_buffer(data, dtype, shape, block_size, out_blocks):
    arr = np.frombuffer(data, dtype=dtype)
    arr = arr.reshape((-1, shape))

    for i in range(len(out_blocks)):
        out_blocks[i] = arr[:, i * block_size:(i + 1) * block_size]


@constraint(computing_units="${ComputingUnits}")
@task(out_blocks=COLLECTION_OUT)
def _read_lines(lines, block_size, delimiter, out_blocks, col_of_index=False):
    samples = np.genfromtxt(lines, delimiter=delimiter)

    if len(samples.shape) == 1:
        samples = samples.reshape(1, -1)

    if col_of_index:
        for i, j in enumerate(range(0, samples.shape[1], block_size)):
            out_blocks[i] = samples[:, j + 1:j + block_size + 1]
    else:
        for i, j in enumerate(range(0, samples.shape[1], block_size)):
            out_blocks[i] = samples[:, j:j + block_size]


@constraint(computing_units="${ComputingUnits}")
@task(out_blocks={Type: COLLECTION_OUT, Depth: 2})
def _read_svmlight(lines, out_blocks, col_size, n_features, store_sparse):
    from tempfile import SpooledTemporaryFile
    from sklearn.datasets import load_svmlight_file

    # Creating a tmp file to use load_svmlight_file method should be more
    # efficient than parsing the lines manually
    tmp_file = SpooledTemporaryFile(mode="wb+", max_size=2e8)
    tmp_file.writelines(lines)
    tmp_file.seek(0)

    x, y = load_svmlight_file(tmp_file, n_features=n_features)
    if not store_sparse:
        x = x.toarray()

    # tried also converting to csc/ndarray first for faster splitting but it's
    # not worth. Position 0 contains the X
    for i in range(ceil(n_features / col_size)):
        out_blocks[0][i] = x[:, i * col_size:(i + 1) * col_size]

    # Position 1 contains the y block
    out_blocks[1][0] = y.reshape(-1, 1)


def _load_blocks_array(blocks, shape, block_size):
    if shape[0] < block_size[0] or shape[1] < block_size[1]:
        raise ValueError("The block size is greater than the ds-array")
    return Array(blocks, shape=shape, top_left_shape=block_size,
                 reg_shape=block_size, sparse=False)


def _load_mdcrd_copy(path, block_size, n_cols, n_hblocks, bytes_per_snap,
                     bytes_per_block):
    file_size = os.stat(path).st_size - _CRD_LINE_SIZE
    blocks = []

    for i in range(0, file_size, bytes_per_block):
        out_blocks = [object() for _ in range(n_hblocks)]
        _read_crd_file(path, i, bytes_per_block, block_size[1], n_cols,
                       out_blocks)
        blocks.append(out_blocks)

    n_samples = int(file_size / bytes_per_snap)

    return Array(blocks, top_left_shape=block_size, reg_shape=block_size,
                 shape=(n_samples, n_cols), sparse=False)


def _load_mdcrd(path, block_size, n_cols, n_blocks, bytes_per_snap,
                bytes_per_block):
    blocks = []

    file_size = os.stat(path).st_size - _CRD_LINE_SIZE

    try:
        fid = open(path, "rb")
        fid.read(_CRD_LINE_SIZE)  # skip header

        for _ in range(0, file_size, bytes_per_block):
            data = fid.read(bytes_per_block)
            out_blocks = [object() for _ in range(n_blocks)]
            _read_crd_bytes(data, block_size[1], n_cols, out_blocks)
            blocks.append(out_blocks)
    finally:
        fid.close()

    n_samples = int(file_size / bytes_per_snap)

    return Array(blocks, top_left_shape=block_size, reg_shape=block_size,
                 shape=(n_samples, n_cols), sparse=False)


@constraint(computing_units="${ComputingUnits}")
@task(data=IN_DELETE, out_blocks=COLLECTION_INOUT)
def _read_crd_bytes(data, hblock_size, n_cols, out_blocks):
    arr = np.fromstring(data.decode(), sep=" ")
    arr = arr.reshape((-1, n_cols))

    for i in range(len(out_blocks)):
        out_blocks[i] = arr[:, i * hblock_size:(i + 1) * hblock_size]


@constraint(computing_units="${ComputingUnits}")
@task(path=FILE_IN, out_blocks=COLLECTION_INOUT)
def _read_crd_file(path, start, read_size, hblock_size, n_cols, out_blocks):
    with open(path, "rb") as fid:
        fid.seek(start + _CRD_LINE_SIZE)  # skip header and go to start
        data = fid.read(read_size)

    arr = np.fromstring(data.decode(), sep=" ")
    arr = arr.reshape((-1, n_cols))

    for i in range(len(out_blocks)):
        out_blocks[i] = arr[:, i * hblock_size:(i + 1) * hblock_size]


@constraint(computing_units="${ComputingUnits}")
@task(block_files=COLLECTION_FILE_IN)
def _load_hstack_npy_block(block_files, start_col, end_col):
    if len(block_files) == 1:
        return np.load(block_files[0])[:, start_col:end_col]
    arrays = [np.load(block_files[0])[:, start_col:]]
    for file in block_files[1:-1]:
        arrays.append(np.load(file))
    arrays.append(np.load(block_files[-1])[:, :end_col])
    return np.concatenate(arrays, axis=1)
