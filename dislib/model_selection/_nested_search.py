import sys

from pycompss.api.api import compss_wait_on
from pycompss.api.compss import compss
from pycompss.api.parameter import FILE_OUT
from pycompss.api.task import task
from pycompss.util.serialization.serializer import serialize_to_file,\
    deserialize_from_file

from dislib.data.array import Array
from dislib.model_selection._validation import fit_and_score


class ArrayInfo:
    def __init__(self, name, n_blocks, constr_params):
        self.name = name
        self.n_blocks = n_blocks
        self.constr_params = constr_params


def disassemble_arrays(data):
    arrays = {'x_train': data[0][0], 'y_train': data[0][1],
              'x_test': data[1][0], 'y_test': data[1][1]}
    args = []
    for name, arr in arrays.items():
        if arr is not None:
            constr_params = {'top_left_shape': arr._top_left_shape,
                             'reg_shape': arr._reg_shape,
                             'shape': arr._shape,
                             'sparse': arr._sparse}
            args.append(ArrayInfo(name, arr._n_blocks, constr_params))
            for row_blocks in arr._blocks:
                args.extend(row_blocks)
    return args


def reassemble_arrays(varargs):
    varargs_it = iter(varargs)
    arrays = {}
    for array_info in varargs_it:
        n_blocks = array_info.n_blocks
        blocks = []
        for i in range(n_blocks[0]):
            blocks.append([])
            for j in range(n_blocks[1]):
                blocks[i].append(next(varargs_it))
        arrays[array_info.name] = Array(blocks, **array_info.constr_params)

    x_train = arrays['x_train']
    y_train = arrays.get('y_train')  # Optional, so it can be None
    x_test = arrays['x_test']
    y_test = arrays.get('y_test')  # Optional, so it can be None

    return (x_train, y_train), (x_test, y_test)


def evaluate_candidate_nested(out, estimator, scorer, params, fit_params,
                              data):
    args = disassemble_arrays(data)
    return nested_fit_and_score(out, estimator, scorer, params, fit_params,
                                *args)


@compss(runcompss="runcompss", flags="-d --python_interpreter=python3",
        app_name=__file__)
@task(out=FILE_OUT, returns=int)
def nested_fit_and_score(out, estimator, scorer, params, fit_params, *varargs):
    pass


def deserialize_args(args):
    for arg in args:
        yield deserialize_from_file(arg)


def main():
    out = sys.argv[1]
    arg_objs = list(deserialize_args(sys.argv[2:]))
    estimator, scorer, params, fit_params = arg_objs[0:4]
    train_ds, test_ds = reassemble_arrays(arg_objs[4:])
    res = compss_wait_on(fit_and_score(estimator, train_ds, test_ds, scorer,
                                       params, fit_params))
    serialize_to_file(res, out)


if __name__ == '__main__':
    main()
