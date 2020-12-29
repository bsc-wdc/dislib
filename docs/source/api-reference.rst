API Reference
=============

dislib.array: Distributed array
-------------------------------

Classes
.......

:class:`data.Array <dislib.data.array.Array>` - 2-dimensional array divided in
blocks that can be operated in a distributed way.


Array creation routines
.......................

:meth:`dislib.array <dislib.array>` - Build a distributed array
(ds-array) from an array-like structure, such as a NumPy array, a list, or a SciPy sparse matrix.

:meth:`dislib.random_array <dislib.random_array>` - Build a ds-array with
random contents.

:meth:`dislib.zeros <dislib.zeros>` - Build a ds-array filled with zeros.

:meth:`dislib.full <dislib.full>` - Build a ds-array filled with a value.

:meth:`dislib.identity <dislib.identity>` - Build an identity ds-array.

:meth:`dislib.load_svmlight_file <dislib.load_svmlight_file>` - Build a
ds-array from a file in `SVMlight <http://svmlight.joachims.org/>`_ format.

:meth:`dislib.load_txt_file <dislib.load_txt_file>` - Build a
ds-array from a text file.

:meth:`dislib.load_npy_file <dislib.load_npy_file>` - Build a ds-array from
a binary NumPy file.

:meth:`dislib.load_mdcrd_file <dislib.load_mdcrd_file>` - Build a ds-array
from a mdcrd trajectory file.


Other functions
---------------

:meth:`dislib.apply_along_axis <dislib.apply_along_axis>` - Applies a
function to a ds-array along a given axis.

dislib.utils: Utility functions
-------------------------------------

:meth:`utils.shuffle <dislib.utils.base.shuffle>` - Randomly shuffles the
rows of a ds-array.

dislib.math: Mathematical functions
-----------------------------------

:meth:`dislib.kron <dislib.kron>` - Computes the Kronecker product of two
ds-arrays.

:meth:`dislib.svd <dislib.svd>` - Singular value decomposition of a ds-array.

dislib.preprocessing: Data pre-processing
-----------------------------------------

:class:`preprocessing.StandardScaler <dislib.preprocessing.classes.StandardScaler>` -
Scale a ds-array to zero mean and unit variance.

dislib.decomposition: Matrix Decomposition
------------------------------------------

:class:`decomposition.PCA <dislib.decomposition.pca.base.PCA>` - Principal
component analysis (PCA).

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


dislib.classification: Classification
-------------------------------------

:class:`classification.CascadeSVM <dislib.classification.csvm.base.CascadeSVM>`
- Distributed support vector classification using a cascade of classifiers.

:class:`classification.RandomForestClassifier <dislib.classification.rf.forest.RandomForestClassifier>` -
Build a random forest for classification.


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


dislib.optimization: Optimization
---------------------------------

:class:`optimization.ADMM <dislib.optimization.admm.base.ADMM>` - Alternating
Direction Method of Multipliers (ADMM) solver.


dislib.neighbors: Neighbor queries
----------------------------------

:class:`cluster.NearestNeighbors <dislib.neighbors.base.NearestNeighbors>` -
Perform k-nearest neighbors queries.


dislib.model_selection: Model selection
---------------------------------------

:class:`model_selection.GridSearchCV <dislib.model_selection.GridSearchCV>` -
Exhaustive search over specified parameter values for an estimator.

:class:`model_selection.RandomizedSearchCV <dislib.model_selection.RandomizedSearchCV>` -
Randomized search over estimator parameters sampled from given distributions.

:class:`model_selection.KFold <dislib.model_selection.KFold>` -
K-fold splitter for cross-validation.
