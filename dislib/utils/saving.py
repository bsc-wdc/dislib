import json
import os
import numpy as np

from pycompss.runtime.management.classes import Future
from pycompss.api.api import compss_wait_on

from sklearn.svm import SVC as SklearnSVC
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier
from sklearn.tree._tree import Tree as SklearnTree
from scipy.sparse import csr_matrix

import dislib as ds
import dislib.classification
import dislib.cluster
import dislib.recommendation
import dislib.regression
from dislib.data.array import Array
from dislib.commons.rf._decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _Node,
    _ClassificationNode,
    _RegressionNode,
    _InnerNodeInfo,
    _LeafInfo,
    _SkTreeWrapper,
)

try:
    import cbor2
except ImportError:
    cbor2 = None

# Dislib models with saving tested (model: str -> module: str)
IMPLEMENTED_MODELS = {
    "KMeans": "cluster",
    "GaussianMixture": "cluster",
    "CascadeSVM": "classification",
    "RandomForestClassifier": "classification",
    "ALS": "recommendation",
    "LinearRegression": "regression",
    "Lasso": "regression",
}

# Classes used by models
DISLIB_CLASSES = {
    "KMeans": dislib.cluster.KMeans,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "_Node": _Node,
    "_ClassificationNode": _ClassificationNode,
    "_RegressionNode": _RegressionNode,
    "_InnerNodeInfo": _InnerNodeInfo,
    "_LeafInfo": _LeafInfo,
    "_SkTreeWrapper": _SkTreeWrapper,
}

SKLEARN_CLASSES = {
    "SVC": SklearnSVC,
    "DecisionTreeClassifier": SklearnDTClassifier,
}


def save_model(model, filepath, overwrite=True, save_format="json"):
    """Saves a model to a file.

    The model is synchronized before saving and can be reinstantiated in the
    exact same state, without any of the code used for model definition or
    fitting.

    Parameters
    ----------
    model : dislib model.
        Dislib model to serialize and save.
    filepath : str
        Path where to save the model
    overwrite : bool, optional (default=True)
        Whether any existing model at the target
        location should be overwritten.
    save_format : str, optional (default='json)
        Format used to save the models.

    Examples
    --------
    >>> from dislib.cluster import KMeans
    >>> from dislib.utils import save_model, load_model
    >>> import numpy as np
    >>> import dislib as ds
    >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> x_train = ds.array(x, (2, 2))
    >>> model = KMeans(n_clusters=2, random_state=0)
    >>> model.fit(x_train)
    >>> save_model(model, '/tmp/model')
    >>> loaded_model = load_model('/tmp/model')
    >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
    >>> model_pred = model.predict(x_test)
    >>> loaded_model_pred = loaded_model.predict(x_test)
    >>> assert np.allclose(model_pred.collect(), loaded_model_pred.collect())
    """

    # Check overwrite
    if not overwrite and os.path.isfile(filepath):
        return

    # Check for dislib model
    model_name = model.__class__.__name__
    if model_name not in IMPLEMENTED_MODELS.keys():
        raise NotImplementedError(
            "Saving has only been implemented for the following models:\n%s"
            % IMPLEMENTED_MODELS.keys()
        )

    # Synchronize model
    if model_name == "RandomForestClassifier":
        _sync_rf(model)

    _sync_obj(model.__dict__)
    model_metadata = model.__dict__.copy()
    model_metadata["model_name"] = model_name

    # Save model
    if save_format == "json":
        with open(filepath, "w") as f:
            json.dump(model_metadata, f, default=_encode_helper)
    elif save_format == "cbor":
        if cbor2 is None:
            raise ModuleNotFoundError("No module named 'cbor2'")
        with open(filepath, "wb") as f:
            cbor2.dump(model_metadata, f, default=_encode_helper_cbor)
    else:
        raise ValueError("Wrong save format.")


def load_model(filepath, load_format="json"):
    """Loads a model from a file.

    The model is reinstantiated in the exact same state in which it was saved,
    without any of the code used for model definition or fitting.

    Parameters
    ----------
    filepath : str
        Path of the saved the model
    load_format : str, optional (default='json')
        Format used to load the model.

    Examples
    --------
    >>> from dislib.cluster import KMeans
    >>> from dislib.utils import save_model, load_model
    >>> import numpy as np
    >>> import dislib as ds
    >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> x_train = ds.array(x, (2, 2))
    >>> model = KMeans(n_clusters=2, random_state=0)
    >>> model.fit(x_train)
    >>> save_model(model, '/tmp/model')
    >>> loaded_model = load_model('/tmp/model')
    >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
    >>> model_pred = model.predict(x_test)
    >>> loaded_model_pred = loaded_model.predict(x_test)
    >>> assert np.allclose(model_pred.collect(), loaded_model_pred.collect())
    """
    # Load model
    if load_format == "json":
        with open(filepath, "r") as f:
            model_metadata = json.load(f, object_hook=_decode_helper)
    elif load_format == "cbor":
        if cbor2 is None:
            raise ModuleNotFoundError("No module named 'cbor2'")
        with open(filepath, "rb") as f:
            model_metadata = cbor2.load(f, object_hook=_decode_helper_cbor)
    else:
        raise ValueError("Wrong load format.")

    # Check for dislib model
    model_name = model_metadata["model_name"]
    if model_name not in IMPLEMENTED_MODELS.keys():
        raise NotImplementedError(
            "Saving has only been implemented for the following models:\n%s"
            % IMPLEMENTED_MODELS.keys()
        )
    del model_metadata["model_name"]

    # Create model
    model_module = getattr(ds, IMPLEMENTED_MODELS[model_name])
    model_class = getattr(model_module, model_name)
    model = model_class()
    model.__dict__.update(model_metadata)

    # Set class methods
    if model_name == "CascadeSVM" and "kernel" in model_metadata:
        try:
            model._kernel_f = getattr(
                model, model._name_to_kernel[model_metadata["kernel"]]
            )
        except AttributeError:
            model._kernel_f = getattr(model, "_rbf_kernel")

    return model


def _encode_helper_cbor(encoder, obj):
    """Special encoder wrapper for dislib using cbor2."""
    encoder.encode(_encode_helper(obj))


def _decode_helper_cbor(decoder, obj):
    """Special decoder wrapper for dislib using cbor2."""
    return _decode_helper(obj)


def _encode_helper(obj):
    """Special encoder for dislib that serializes the different objectes
    and stores their state for future loading.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, range):
        return {
            "class_name": "range",
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        }
    elif isinstance(obj, csr_matrix):
        return {
            "class_name": "csr_matrix",
            **obj.__dict__,
        }
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
    elif callable(obj):
        return {
            "class_name": "callable",
            "module": obj.__module__,
            "name": obj.__name__,
        }
    elif isinstance(obj, SklearnTree):
        return {
            "class_name": obj.__class__.__name__,
            "n_features": obj.n_features,
            "n_classes": obj.n_classes,
            "n_outputs": obj.n_outputs,
            "items": obj.__getstate__(),
        }
    elif isinstance(
        obj, tuple(DISLIB_CLASSES.values()) + tuple(SKLEARN_CLASSES.values())
    ):
        return {
            "class_name": obj.__class__.__name__,
            "module_name": obj.__module__,
            "items": obj.__dict__,
        }
    raise TypeError("Not JSON Serializable:", obj)


def _decode_helper(obj):
    """Special decoder for dislib that instantiates the different objects
    and updates their attributes to recover the saved state.
    """
    if isinstance(obj, dict) and "class_name" in obj:

        class_name = obj["class_name"]
        if class_name == "range":
            return range(obj["start"], obj["stop"], obj["step"])
        elif class_name == "tuple":
            return tuple(obj["items"])
        elif class_name == "ndarray":
            if obj["dtype_list"]:
                items = list(map(tuple, obj["items"]))
                return np.rec.fromrecords(items, dtype=eval(obj["dtype"]))
            else:
                return np.array(obj["items"], dtype=obj["dtype"])
        elif class_name == "csr_matrix":
            return csr_matrix(
                (obj["data"], obj["indices"], obj["indptr"]),
                shape=obj["_shape"],
            )
        elif class_name == "dsarray":
            return Array(
                blocks=obj["_blocks"],
                top_left_shape=obj["_top_left_shape"],
                reg_shape=obj["_reg_shape"],
                shape=obj["_shape"],
                sparse=obj["_sparse"],
                delete=obj["_delete"],
            )
        elif class_name == "RandomState":
            random_state = np.random.RandomState()
            random_state.set_state(_decode_helper(obj["items"]))
            return random_state
        elif class_name == "Tree":
            dict_ = _decode_helper(obj["items"])
            model = SklearnTree(
                obj["n_features"], obj["n_classes"], obj["n_outputs"]
            )
            model.__setstate__(dict_)
            return model
        elif (
            class_name in DISLIB_CLASSES.keys()
            and "dislib" in obj["module_name"]
        ):
            dict_ = _decode_helper(obj["items"])
            if class_name == "DecisionTreeClassifier":
                model = DISLIB_CLASSES[obj["class_name"]](
                    try_features=dict_.pop("try_features"),
                    max_depth=dict_.pop("max_depth"),
                    distr_depth=dict_.pop("distr_depth"),
                    sklearn_max=dict_.pop("sklearn_max"),
                    bootstrap=dict_.pop("bootstrap"),
                    random_state=dict_.pop("random_state"),
                )
            elif class_name == "_SkTreeWrapper":
                sk_tree = _decode_helper(dict_.pop("sk_tree"))
                model = DISLIB_CLASSES[obj["class_name"]](sk_tree)
            else:
                model = DISLIB_CLASSES[obj["class_name"]]()
            model.__dict__.update(dict_)
            return model
        elif (
            class_name in SKLEARN_CLASSES.keys()
            and "sklearn" in obj["module_name"]
        ):
            dict_ = _decode_helper(obj["items"])
            model = SKLEARN_CLASSES[obj["class_name"]]()
            model.__dict__.update(dict_)
            return model
        elif class_name == "callable":
            if obj["module"] == "numpy":
                return getattr(np, obj["name"])
            return None

    return obj


def _sync_obj(obj):
    """Recursively synchronizes the Future objects of a list or dictionary
    by using `compss_wait_on(obj)`.
    """
    if isinstance(obj, dict):
        iterator = iter(obj.items())
    elif isinstance(obj, list):
        iterator = iter(enumerate(obj))
    else:
        print(obj)
        raise ValueError("Expected dict or list and received %s." % type(obj))

    for key, val in iterator:
        if isinstance(val, (dict, list)):
            _sync_obj(obj[key])
        else:
            obj[key] = compss_wait_on(val)
            if isinstance(obj[key], Future):
                raise TypeError(
                    "Could not synchronize Future (%s, %s)." % (key, val)
                )
            if isinstance(getattr(obj[key], "__dict__", None), dict):
                _sync_obj(obj[key].__dict__)


def _sync_rf(rf):
    """Sync the `try_features` and `n_classes` attribute of the different trees
    since they cannot be synced recursively.
    """
    if isinstance(rf.trees[0].try_features, Future):
        try_features = compss_wait_on(rf.trees[0].try_features)
        n_classes = compss_wait_on(rf.trees[0].n_classes)
        for tree in rf.trees:
            tree.try_features = try_features
            tree.n_classes = n_classes
