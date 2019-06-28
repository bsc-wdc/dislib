# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Removed

[Unreleased]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/bsc-wdc/dislib/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bsc-wdc/dislib/compare/v0.1.0...v0.2.0

