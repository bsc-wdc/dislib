API Reference
=============

dislib.data: Data handling utilities
------------------------------------

Classes
.......

:class:`data.Dataset <dislib.data.classes.Dataset>` - Main data structure for
handling distributed datasets. Dataset works as a list of Subset.

:class:`data.Subset <dislib.data.classes.Subset>` - Collection of samples and
(optionally) labels.


Functions
.........

:meth:`data.load_data <dislib.data.base.load_data>` - Build a
:class:`Dataset <dislib.data.classes.Dataset>` from an ndarray.

:meth:`data.load_libsvm_file <dislib.data.base.load_libsvm_file>` - Build a
:class:`Dataset <dislib.data.classes.Dataset>` from a file in LibSVM format
(sparse).

:meth:`data.load_libsvm_files <dislib.data.base.load_libsvm_files>` - Build a
:class:`Dataset <dislib.data.classes.Dataset>` from multiple files in LibSVM
format (sparse).

:meth:`data.load_txt_file <dislib.data.base.load_txt_file>` - Build a
:class:`Dataset <dislib.data.classes.Dataset>` from a text file.

:meth:`data.load_txt_files <dislib.data.base.load_txt_files>` - Build a
:class:`Dataset <dislib.data.classes.Dataset>` from multiple text files.


dislib.utils: Other utility functions
-------------------------------------

:meth:`utils.as_grid <dislib.utils.base.as_grid>` - Re-organizes samples in a
:class:`Dataset <dislib.data.classes.Dataset>`
in a hyper-dimensional grid, where each
:class:`Subset <dislib.data.classes.Subset>` represents a region in this space.

:meth:`utils.shuffle <dislib.utils.base.shuffle>` - Randomly shuffles the
samples in a :class:`Dataset <dislib.data.classes.Dataset>`.


dislib.preprocessing: Data pre-processing
-----------------------------------------

Classes
.......

:class:`preprocessing.StandardScaler <dislib.preprocessing.classes.StandardScaler>` -
Scale data to zero mean and unit variance.

dislib.cluster: Clustering
--------------------------

Classes
.......

:class:`cluster.DBSCAN <dislib.cluster.dbscan.base.DBSCAN>` - Perform DBSCAN
clustering.

:class:`cluster.KMeans <dislib.cluster.kmeans.base.KMeans>` - Perform K-Means
clustering.

:class:`cluster.GaussianMixture <dislib.cluster.gm.base.GaussianMixture>` -
Fit a gaussian mixture model.


dislib.classification: Classification
-------------------------------------

Classes
.......

:class:`classification.CascadeSVM <dislib.classification.csvm.base.CascadeSVM>`
- Distributed support vector classification using a cascade of classifiers.

:class:`classification.RandomForestClassifier <dislib.classification.rf.forest.RandomForestClassifier>` -
Build a random forest for classification.


dislib.neighbors: Neighbor queries
------------------------------------

Classes
-------

:class:`cluster.NearestNeighbors <dislib.neighbors.base.NearestNeighbors>` -
Perform k-nearest neighbors queries.

Other functions
---------------

:meth:`fft <dislib.fft.base.fft>` - Distributed fast fourier transform
computation.


