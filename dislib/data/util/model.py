import numpy as np
from dislib.data.array import Array

from pycompss.api.api import compss_wait_on

try:
    import cbor2
except ImportError:
    cbor2 = None

try:
    import blosc2
except ImportError:
    blosc2 = None


def encoder_helper(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return {
            "class_name": "ndarray",
            "dtype_list": len(obj.dtype.descr) > 1,
            "dtype": str(obj.dtype),
            "items": obj.tolist(),
        }
    elif isinstance(obj, Array):
        return {"class_name": "dsarray", **obj.__dict__}
    elif isinstance(obj, np.random.RandomState):
        return {"class_name": "RandomState", "items": obj.get_state()}
    return None


def decoder_helper(class_name, obj):
    if class_name == "ndarray":
        if obj["dtype_list"]:
            items = list(map(tuple, obj["items"]))
            return np.rec.fromrecords(items, dtype=eval(obj["dtype"]))
        else:
            return np.array(obj["items"], dtype=obj["dtype"])
    elif class_name == "dsarray":
        return Array(
            blocks=obj["_blocks"],
            top_left_shape=obj["_top_left_shape"],
            reg_shape=obj["_reg_shape"],
            shape=obj["_shape"],
            sparse=obj["_sparse"],
            delete=obj["_delete"],
        )
    return None


def sync_obj(obj):
    """Recursively synchronizes the Future objects of a list or dictionary
    by using `compss_wait_on(obj)`.
        """
    if isinstance(obj, dict):
        iterator = iter(obj.items())
    elif isinstance(obj, list):
        iterator = iter(enumerate(obj))
    else:
        raise TypeError("Expected dict or list and received %s." % type(obj))

    for key, val in iterator:
        if isinstance(val, (dict, list)):
            sync_obj(obj[key])
        else:
            obj[key] = compss_wait_on(val)
            if isinstance(getattr(obj[key], "__dict__", None), dict):
                sync_obj(obj[key].__dict__)
