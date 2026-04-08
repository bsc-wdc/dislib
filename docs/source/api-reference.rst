API Reference
=============

.. toctree::
   :hidden:
   :caption: Arrays

   dislib.array
   dislib.tensor
   dislib.data.util

.. toctree::
   :hidden:
   :caption: Classification

   dislib.classification.csvm
   dislib.classification.knn

.. toctree::
   :hidden:
   :caption: Clustering

   dislib.cluster.daura
   dislib.cluster.dbscan
   dislib.cluster.gm
   dislib.cluster.kmeans

.. toctree::
   :hidden:
   :caption: Decomposition

   dislib.decomposition

.. toctree::
   :hidden:
   :caption: Math

   dislib.math

.. toctree::
   :hidden:
   :caption: Neural Networks

   dislib.pytorch
   dislib.eddl

.. toctree::
   :hidden:
   :caption: Model Selection

   dislib.model_selection

.. toctree::
   :hidden:
   :caption: Neighbors

   dislib.neighbors

.. toctree::
   :hidden:
   :caption: Optimization

   dislib.optimization.admm

.. toctree::
   :hidden:
   :caption: Preprocessing

   dislib.preprocessing

.. toctree::
   :hidden:
   :caption: Recommendation

   dislib.recommendation.als

.. toctree::
   :hidden:
   :caption: Regression

   dislib.regression.lasso
   dislib.regression.linear

.. toctree::
   :hidden:
   :caption: Sorting

   dislib.sorting.terasort

.. toctree::
   :hidden:
   :caption: Trees

   dislib.trees
   dislib.trees.distributed
   dislib.trees.mmap
   dislib.trees.nested

.. toctree::
   :hidden:
   :caption: Utilities

   dislib.utils

dislib.array: Distributed array
-------------------------------

Classes
.......

:class:`data.Array <dislib.data.array.Array>` - 2-dimensional array divided in
blocks that can be operated in a distributed way.

:class:`data.Tensor <dislib.data.tensor.Tensor>` - n-dimensional tensor divided in
blocks that can be operated in a distributed way.


Array creation routines
.......................

:meth:`dislib.array <dislib.array>` - Build a distributed array
(ds-array) from an array-like structure, such as a NumPy array, a list, or a SciPy sparse matrix.

:meth:`dislib.random_array <dislib.random_array>` - Build a ds-array with
random contents.

:meth:`dislib.zeros <dislib.zeros>` - Build a ds-array filled with zeros.

:meth:`dislib.full <dislib.full>` - Build a ds-array filled with a value.

:meth:`dislib.eye <dislib.eye>` - Build an eye ds-array.

:meth:`dislib.identity <dislib.identity>` - Build an identity ds-array.

:meth:`dislib.load_svmlight_file <dislib.load_svmlight_file>` - Build a
ds-array from a file in `SVMlight <http://svmlight.joachims.org/>`_ format.

:meth:`dislib.load_txt_file <dislib.load_txt_file>` - Build a
ds-array from a text file.

:meth:`dislib.load_npy_file <dislib.load_npy_file>` - Build a ds-array from
a binary NumPy file.

:meth:`dislib.load_mdcrd_file <dislib.load_mdcrd_file>` - Build a ds-array
from a mdcrd trajectory file.

:meth:`dislib.data.load_hstack_npy_files <dislib.data.load_hstack_npy_files>` - Build a ds-array
from .npy files, concatenating them side-by-side.

:meth:`dislib.save_txt <dislib.save_txt>` - Save a ds-array by blocks to a
directory in txt format.

Utility functions
.......................

:meth:`data.util.compute_bottom_right_shape <dislib.data.util.compute_bottom_right_shape>` -
Computes a shape of the bottom right block.

:meth:`data.util.pad <dislib.data.util.pad>` - Pad array blocks with
the desired value.

:meth:`data.util.pad_last_blocks_with_zeros <dislib.data.util.pad_last_blocks_with_zeros>` -
Pad array blocks with zeros.

:meth:`data.util.remove_last_columns <dislib.data.util.remove_last_columns>` -
Removes last columns from the right-most blocks of the ds-array.

:meth:`data.util.remove_last_rows <dislib.data.util.remove_last_rows>` -
Removes last rows from the bottom blocks of the ds-array.

Tensor creation routines
........................

:meth:`dislib.from_array <dislib.from_array>` - Build a ds-tensor from a NumPy array.

:meth:`dislib.from_pt_tensor <dislib.from_pt_tensor>` - Build a ds-tensor from a PyTorch tensor.

:meth:`dislib.from_ds_array <dislib.from_ds_array>` - Build a ds-tensor from a ds-array.

:meth:`dislib.create_ds_tensor <dislib.create_ds_tensor>` - Build a ds-tensor from a list of tensors.

:meth:`dislib.random_tensors <dislib.random_tensors>` - Build a ds-tensor with random contents.

:meth:`dislib.data.tensor.load_dataset <dislib.data.tensor.load_dataset>` - Build a ds-tensor from a dataset of files.


Tensor utility functions
........................

:meth:`dislib.data.tensor.cat <dislib.data.tensor.cat>` - Concatenate ds-tensors along a dimension.

:meth:`dislib.data.tensor.change_shape <dislib.data.tensor.change_shape>` - Change the shape of a ds-tensor.

:meth:`dislib.data.tensor.rechunk_tensor <dislib.data.tensor.rechunk_tensor>` - Rechunk a ds-tensor along a dimension.

:meth:`dislib.data.tensor.shuffle <dislib.data.tensor.shuffle>` - Randomly shuffle a ds-tensor.


Other functions
...............

:meth:`dislib.apply_along_axis <dislib.apply_along_axis>` - Applies a
function to a ds-array along a given axis.


dislib.classification: Classification
-------------------------------------

:class:`classification.CascadeSVM <dislib.classification.csvm.base.CascadeSVM>`
- Distributed support vector classification using a cascade of classifiers.

:class:`classification.KNeighborsClassifier <dislib.classification.knn.base.KNeighborsClassifier>`
- Distributed K neighbors classification using partial classifiers.


dislib.cluster: Clustering
--------------------------

:class:`cluster.DBSCAN <dislib.cluster.dbscan.base.DBSCAN>` - Perform DBSCAN
clustering.

:class:`cluster.KMeans <dislib.cluster.kmeans.base.KMeans>` - Perform K-Means
clustering.

:class:`cluster.GaussianMixture <dislib.cluster.gm.base.GaussianMixture>` -
Fit a gaussian mixture model.

:class:`cluster.Daura <dislib.cluster.daura.base.Daura>` - Perform Daura
clustering.


dislib.decomposition: Matrix Decomposition
------------------------------------------

:meth:`decomposition.qr <dislib.decomposition.qr.base.qr>` -
QR decomposition.

:class:`decomposition.tsqr <dislib.decomposition.tsqr.base.tsqr>` -
Tall-Skinny QR decomposition.

:class:`decomposition.PCA <dislib.decomposition.pca.base.PCA>` -
Principal
Component Analysis (PCA).

:class:`decomposition.lanczos_svd <dislib.decomposition.lanczos.base.lanczos_svd>` -
Lanczos SVD decomposition.

:class:`decomposition.random_svd <dislib.decomposition.randomsvd.base.random_svd>` -
Random SVD decomposition.


dislib.math: Mathematical functions
-----------------------------------

:meth:`dislib.kron <dislib.kron>` - Computes the Kronecker product of two
ds-arrays.

:meth:`dislib.svd <dislib.svd>` - Singular value decomposition of a ds-array.

dislib.pytorch and dislib.eddl: Distributed neural network training
--------------------------------------------------------------------

:class:`pytorch.EncapsulatedFunctionsDistributedPytorch <dislib.pytorch.encapsulated_functions_distributed.EncapsulatedFunctionsDistributedPytorch>` - Distributed training of neural networks using PyTorch backend.

:class:`eddl.EncapsulatedFunctionsDistributedEddl <dislib.eddl.encapsulate_function_distributed.EncapsulatedFunctionsDistributedEddl>` - Distributed training of neural networks using EDDL backend.

dislib.model_selection: Model selection
---------------------------------------

:class:`model_selection.GridSearchCV <dislib.model_selection.GridSearchCV>` -
Exhaustive search over specified parameter values for an estimator.

:class:`model_selection.RandomizedSearchCV <dislib.model_selection.RandomizedSearchCV>` -
Randomized search over estimator parameters sampled from given distributions.

:class:`model_selection.SimulationGridSearch <dislib.model_selection.SimulationGridSearch>` -
Exhaustive search over estimator parameters sampled from given distributions.

:class:`model_selection.KFold <dislib.model_selection.KFold>` -
K-fold splitter for cross-validation.

:meth:`model_selection.train_test_split <dislib.model_selection.train_test_split>` -
Split arrays or matrices into random train and test subsets.


dislib.neighbors: Neighbor queries
----------------------------------

:class:`cluster.NearestNeighbors <dislib.neighbors.base.NearestNeighbors>` -
Perform k-nearest neighbors queries.


dislib.preprocessing: Data pre-processing
-----------------------------------------

:class:`preprocessing.MinMaxScaler <dislib.preprocessing.MinMaxScaler>` -
Scale a ds-array to zero mean and unit variance.

:class:`preprocessing.StandardScaler <dislib.preprocessing.StandardScaler>` -
Scale a ds-array to the given range.


dislib.recommendation: Recommendation
-------------------------------------

:class:`recommendation.ALS <dislib.recommendation.als.base.ALS>`
- Distributed alternating least squares for collaborative filtering.


dislib.regression: Regression
-----------------------------

:class:`regression.LinearRegression <dislib.regression.linear.base.LinearRegression>`
- Multivariate linear regression using ordinary least squares.


:class:`regression.Lasso <dislib.regression.lasso.base.Lasso>`
- Linear Model trained with L1 prior as regularizer.


dislib.sorting: Sorting
-----------------------------

:class:`sorting.TeraSort <dislib.sorting.terasort.base.TeraSort>`
-  Sorts the ds-array using the TeraSort algorithm.

dislib.trees: Trees
-------------------------------------

:class:`trees.DecisionTreeClassifier <dislib.trees.DecisionTreeClassifier>` -
Build a decision tree.

:class:`trees.DecisionTreeRegressor <dislib.trees.DecisionTreeRegressor>` -
Build a regression tree.

:class:`trees.RandomForestClassifier <dislib.trees.RandomForestClassifier>` -
Build a random forest for classification.

:class:`trees.RandomForestRegressor <dislib.trees.RandomForestClassifier>` -
Build a random forest for regression.

:class:`trees.mmap.DecisionTreeClassifier <dislib.trees.mmap.DecisionTreeClassifier>` -
Build a decision tree using memorymap.

:class:`trees.mmap.DecisionTreeRegressor <dislib.trees.mmap.DecisionTreeRegressor>` -
Build a regression tree using memorymap.

:class:`trees.mmap.RandomForestClassifier <dislib.trees.mmap.RandomForestClassifier>` -
Build a random forest for classification using memorymap.

:class:`trees.mmap.RandomForestRegressor <dislib.trees.mmap.RandomForestClassifier>` -
Build a random forest for regression using memorymap.

:class:`trees.distributed.DecisionTreeClassifier <dislib.trees.distributed.DecisionTreeClassifier>` -
Build a decision tree using the distributed approach.

:class:`trees.distributed.DecisionTreeRegressor <dislib.trees.distributed.DecisionTreeRegressor>` -
Build a regression tree using the distributed approach.

:class:`trees.distributed.RandomForestClassifier <dislib.trees.distributed.RandomForestClassifier>` -
Build a random forest for classification using the distributed approach.

:class:`trees.distributed.RandomForestRegressor <dislib.trees.distributed.RandomForestClassifier>` -
Build a random forest for regression using the distributed approach.

:class:`trees.nested.DecisionTreeClassifier <dislib.trees.nested.DecisionTreeClassifier>` -
Build a decision tree using the nested approach.

:class:`trees.nested.DecisionTreeRegressor <dislib.trees.nested.DecisionTreeRegressor>` -
Build a regression tree using the nested approach.

:class:`trees.nested.RandomForestClassifier <dislib.trees.nested.RandomForestClassifier>` -
Build a random forest for classification using the nested approach.

:class:`trees.nested.RandomForestRegressor <dislib.trees.nested.RandomForestClassifier>` -
Build a random forest for regression using the nested approach.


dislib.utils: Utility functions
-------------------------------------

:meth:`utils.shuffle <dislib.utils.base.shuffle>` - Randomly shuffles the
rows of a ds-array.
