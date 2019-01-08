API Reference
=============

dislib.data: Data handling utilities
------------------------------------

Classes
.......

:class:`data.Dataset <dislib.data.classes.Dataset>`     Main data structure for
handling distributed datasets. Dataset works as a list of Subset.

:class:`data.Subset <dislib.data.classes.Subset>`       Collection of samples
and (optionally) labels.


Functions
.........

:meth:`data.load_data <dislib.data.base.load_data>`

:meth:`data.load_libsvm_file <dislib.data.base.load_libsvm_file>`

:meth:`data.load_libsvm_files <dislib.data.base.load_libsvm_files>`

:meth:`data.load_libsvm_file <dislib.data.base.load_csv_file>`

:meth:`data.load_libsvm_files <dislib.data.base.load_csv_files>`


dislib.utils: Other utility functions
-------------------------------------

:meth:`utils.as_grid <dislib.utils.base.as_grid>`

:meth:`utils.shuffle <dislib.utils.base.shuffle>`


dislib.cluster: Clustering
--------------------------

Classes
.......

:class:`cluster.DBSCAN <dislib.cluster.dbscan.base.DBSCAN>`

:class:`cluster.KMeans <dislib.cluster.kmeans.base.KMeans>`

:class:`cluster.KMedoids <dislib.cluster.kmedoids.base.KMedoids>`


dislib.classification: Classification
-------------------------------------

Classes
.......

:class:`classification.CascadeSVM <dislib.classification.csvm.base.CascadeSVM>`

:class:`classification.RandomForestClassifier <dislib.classification.rf.forest.RandomForestClassifier>`


Other functions
---------------

:meth:`fft <dislib.fft.base.fft>`


