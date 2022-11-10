# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2022-11-11
### Added
- `save` and `load` methods for all models
- Adding Muliclass CSVM
- Adding TS-QR (Tall Skinny QR)
- New in-place operations for ds-arrays:
  `add` `iadd` `isub`
- Matrix-Subtraction and Matrix-Addition
- Concatenating two ds-arrays by columns
- Save ds-array to `npy` file
- Load ds-array from several `npy` files
- Create ds-arrays from blocks
- GridSearch for simulations & improvements
- Inverse transformation in Scalers
- Train-Test-Split functionality
- Add KNN Classifier
- Better SVD columns pairing
- GPU Support using CUDA/CuPy for algorithms: Kmeans, KNN, SVD, PCA, Matmul, Addition, Subtraction, QR, Kronecker

### Changed
- New documentation for GPU, RandomForest, Scalers

### Fixed
- Fix bug Scalers & tests

## [0.7.0] - 2021-11-10
### Added
- New decomposition algorithm QR
- New preprocessing algorithm MinMaxScaler
- Jenkinsfile for CI automated tests
- ds-array matrix multiplication (matmul)
- New function for ds-array creation
- Add `@constraint(computing_units="${ComputingUnits}")` to all tasks
- More I/O functions for reading and writing ds-arrays
- More tests

### Changed
- Move RandomForest from 'classification' to 'trees'

### Fixed
- Some bugs in the ds-array

## [0.6.0] - 2020-10-09
### Added
- User guide and glossary
- Method to read from npy files
- Support for one-dimensional data in ds-array
- Parametrized ds-array tests
- identity, full and zeros methods that generate ds-arrays filled with a value
- ds-array operators: subtraction, division, conjugate, transpose, item setting, etc.
- matmul, kronecker product and rechunk methods for of ds-arrays
- Automatic deletion of ds-arrays when the GC is called
- Multivariate linear regression
- SVD (Singular Value Decomposition)
- PCA using SVD
- ADMM Lasso algorithm
- Daura clustering algorithm

### Changed
- Improved performance testing scripts and added new tests
- Allow executing applications with params using dislib exec
- Extended and improved the tutorial notebook
- Moved data loading routines to a different file as array.py was getting too big
- apply_along_axis for sparse data now returns sparse ds-arrays
- Updated dislib-base docker image
- Replaced COLLECTION_INOUT parameters with COLLECTION_OUT when possible for improving performance
- Updated requirement PyCOMPSs >= 2.7

### Fixed
- Some bugs in the ds-array
- Internal inconsistencies in transformed_array of PCA

## [0.5.0] - 2019-11-25
### Added
- Grid search and randomized search with cross-validation
- K-fold splitter
- Support for jupyter-notebooks from dislib docker image
- Automatic installation of dislib executable when running pip install
  dislib
- Support for sparse data in PCA
- A new notebook with more usage examples
- jupyter command to dislib executable
- Pointer to sklearn license in LICENSE file
- NOTICE file

### Changed
- Estimators now extend sklearn BaseEstimator
- Extended tutorial notebook with other examples
- Added acknowledgements to README

### Removed
- Pandas dependency in test_als
- CODEOWNERS file

### Fixed
- Small fixes to tutorial notebook
- Small fixes to documentation
- dislib executable now works even if PyCOMPSs is not installed
- Bug fix in ALS performance test
- Several bugs in fancy indexing of ds-arrays
- Fixed dislib executable on MacOS

## [0.4.0] - 2019-09-16
### Added
- Distributed array data structure
- A basic tutorial notebook

### Changed
- Updated docker image to PyCOMPSs 2.5
- Modified the whole library to use distributed arrays instead of Datasets
(including estimators, examples, etc.)
- Added 'init' parameter to K-means
- Updated the developer guide

### Removed
- Dataset and Subset data structures
- FFT estimator
- Methods to load from multiple files

### Fixed
- Fixed the usage of random state in K-means
- Some issues in the performance tests
- Other minor bug fixes

## [0.3.0] - 2019-06-28
### Added
- The VERSION file
- Test for duplicate support vectors in CSVM
- Test for GaussianMixture with random initialization
- New types of covariances for GaussianMixture and more tests
- Scripts for automated performance tests on MareNostrum 4
- A small Performance section to the docs
- Two new algorithms: PCA and LinearRegression
- Added some tests for DBSCAN

### Changed
- Dataset now does not check for duplicate samples (and does not build an 
array of unique IDs). This improves performance signifcantly.
- CSVM now checks and removes duplicate samples generated during the fit 
process.
- GaussianMixture now works with sparse data
- GaussianMixture now removes partial results using compss_delete
- Improved the performance of K-means' _partial_sum task
- Improved docs of GaussianMixture and simplified the code
- Added a check_convergence argument to GaussianMixture
- Significant performance improvement of DBSCAN
- Improved the performance of the shuffle method by using PyCOMPSs COLLECTIONS

### Fixed
- A bug in DBSCAN that was generating incorrect results in certain cases

## [0.2.0] - 2019-03-01
### Added
- This CHANGELOG file
- Added badges to README file
- Added tests for C-SVM and K-means
- Created a utils module with shuffle and as_grid methods
- Added an API reference to the documentation
- Dataset.samples and Dataset.labels properties
- New tests for DBSCAN
- A first version of nearest neighbors algorithm
- Added tests for C-SVM, K-means and DBSCAN with sparse data
- Created a setup.py file and a pip package
- First implementation of Gaussian mixtures and ALS
- Implemented a StandardScaler class as part of a new preprocessing module
- Created a resample method in the utils module
- Dataset transpose
- Dataset apply function

### Changed
- Refactored DBSCAN completely to make code more legible and fix several bugs
- Fixed DBSCAN because it was producing wrong results in some scenarios. Changed the use of disjoint sets to connected components.
- Extended the installation instructions in the README file
- The script classifier_comparison.py now includes Random Forest classifier
- Tests are split into modules
- The COMPSs docker image has been reworked
- Changed the way random_state is used in the different algorithms to ensure proper randomization and reproducibility of different executions.
- Unified the signatures of the different algorithms to fit, predict, and fit_predict. These methods now have the same arguments in all the algorithms.
- Changed license to Apache v2
- Fixed some typos in README
- load methods in the data module can take a delimiter argument now
- Moved the quickstart guide to a separate file and included it in the documentation
- Fixed several bugs

[Unreleased]: https://github.com/bsc-wdc/dislib/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/bsc-wdc/dislib/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bsc-wdc/dislib/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/bsc-wdc/dislib/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bsc-wdc/dislib/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bsc-wdc/dislib/compare/v0.1.0...v0.2.0

