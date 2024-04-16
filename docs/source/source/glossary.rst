Glossary of terms
=================

This page lists alphabetically the meaning of various terms used in
dislib's documentation:

.. these terms can be referenced across the documentation using the
:term: environment

.. glossary::

    array-like
      an instance of an object that can be interpreted as an array (e.g., a
      list, a sparse matrix, a NumPy array, etc.)

    block
      a part of a ds-array that is normally stored remotely.

    column
      a column in a ds-array.

    column block
      a block or set of blocks representing a set of columns in a ds-array.

    csr_matrix
      an instance of a sparse matrix in compressed sparse row format.

    ds-array
      an instance of a distributed array.

    estimator
      anything that learns from data. Typically, an object that fits a model
      given some parameters and input data.

    feature
      each of the dimensions of a sample array. For example, petal
      length or color.

    fit
      learn a model from input data.

    label
      a number that represents the category of a sample.

    ndarray
      an instance of a NumPy array.

    predict
      infer the category of unlabeled data according to a fitted model.

    row
      a row in a ds-array (also a sample).

    row block
      a block or set of blocks that represent a set of rows in a ds-array.

    sample
      an array that normally represents an observation or an instance.
      For example, the characteristics of a particular flower.

    shape
      the total number of rows and columns of a ds-array (or NumPy array)

    spmatrix
      an instance of a SciPy's sparse matrix.

    synchronization
      when the execution of a parallel application stalls until certain data is
      generated, or until all tasks have finished.

    task
      a unit of computation that can be executed in a remote computer.
