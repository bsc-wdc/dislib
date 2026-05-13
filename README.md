<h1 align="center">
    <img src="https://github.com/bsc-wdc/dislib/raw/master/docs/logos/dislib-logo-full.png"
         alt="The Distributed Computing Library"
         height="90px">
</h1>

<h3 align="center">Distributed computing library implemented over PyCOMPSs programming model for HPC.</h3>

<p align="center">
  <a href="https://dislib.bsc.es/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/dislib/badge/?version=stable"
         alt="Documentation Status"/>
  </a>
  <a href="https://github.com/bsc-wdc/dislib">
    <img src="https://compss.bsc.es/jenkins/buildStatus/icon?job=dislib_multibranch%2Fmaster"
         alt="Build Status">
  </a>
  <a href="https://codecov.io/gh/bsc-wdc/dislib">
    <img src="https://codecov.io/gh/bsc-wdc/dislib/branch/master/graph/badge.svg"
         alt="Code Coverage"/>
  </a>
  <a href="https://badge.fury.io/py/dislib">
      <img src="https://badge.fury.io/py/dislib.svg" alt="PyPI version" height="18">
  </a>
  <a href="https://badge.fury.io/py/dislib">
      <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python version" height="18">
  </a>
</p>

<p align="center"><b>
    <a href="https://dislib.bsc.es">Website</a> •
    <a href="https://dislib.bsc.es/en/stable/api-reference.html">Documentation</a> •
    <a href="https://github.com/bsc-wdc/dislib/releases">Releases</a> •
    <a href="mailto:support-compss@bsc.es?subject=Request%20to%20join%20Workflows%20and%20Distributed%20Computing%20Slack&body=Hello%2C%0D%0A%0D%0AI%20would%20like%20to%20request%20access%20to%20the%20Workflows%20and%20Distributed%20Computing%20Slack%20workspace.%20In%20particular%2C%20could%20you%20please%20add%20me%20to%20the%20dislib%20channel%20so%20I%20can%20ask%20for%20support%20and%20stay%20informed%20about%20important%20announcements%3F%0D%0A%0D%0AThank%20you%20very%20much.">Slack</a>
</b></p>

## Introduction

The Distributed Computing Library (dislib) provides distributed algorithms ready to use as a library. So far, dislib is highly focused on machine learning algorithms, and it is greatly inspired by [scikit-learn](https://scikit-learn.org/) and, more recently, by [PyTorch](https://pytorch.org/). However, other types of numerical algorithms might be added in the future. The library has been implemented on top of [PyCOMPSs programming model](http://compss.bsc.es), and it is being developed by the [Workflows and Distributed Computing group](https://github.com/bsc-wdc) of the [Barcelona Supercomputing Center](https://www.bsc.es/). dislib allows easy local development through docker. Once the code is finished, it can be run directly on any distributed platform without any further changes. This includes clusters, supercomputers, clouds, and containerized platforms.

## Contents

- [Introduction](#introduction)
- [Contents](#contents)
- [Quickstart](#quickstart)
- [Availability](#availability)
- [Contributing](#contributing)
- [Citing dislib](#citing-dislib)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Quickstart

Get started with dislib following our [quickstart guide](https://github.com/bsc-wdc/dislib/blob/master/QUICKSTART.md).

## Availability

Currently, the following supercomputers already have PyCOMPSs installed and ready to use. If you need help configuring your own cluster or supercomputer, drop us an email and we will be pleased to help.

- MareNostrum 5 - Barcelona Supercomputing Center (BSC)
- MareNostrum Ona - Barcelona Supercomputing Center (BSC)
- Nord4 - Barcelona Supercomputing Center (BSC)
- JURECA - Jülich Supercomputing Centre (JSC)
- JUSUF - Jülich Supercomputing Centre (JSC)
- JUWELS - Jülich Supercomputing Centre (JSC)
- Barbora - IT4Innovations National Supercomputing Center (IT4I)
- Karolina - IT4Innovations National Supercomputing Center (IT4I)
- Galileo100 - CINECA (CINECA)
- Leonardo - CINECA (CINECA)
- Dardel - PDC Center for High-Performance Computing, KTH (PDC)
- Fugaku - RIKEN Center for Computational Science (R-CCS)
- Hawk - High-Performance Computing Center Stuttgart (HLRS)
- Irene - CEA/TGCC (TGCC)
- Levante - German Climate Computing Centre (DKRZ)
- Mahti - CSC - IT Center for Science (CSC)
- NextGenIO Prototype - EPCC, University of Edinburgh (EPCC)
- Shaheen - King Abdullah University of Science and Technology (KAUST)
- Snellius - SURF (SURF)
- Tirant - University of Valencia (UV)
- Frontier - Oak Ridge National Laboratory (ORNL)
- Jean Zay - IDRIS, CNRS (IDRIS)
- Qmio - Galicia Supercomputing Center (CESGA)

Supported architectures:
- [Intel SSF architectures](https://www.intel.com/content/www/us/en/high-performance-computing/ssf-architecture-specification.html)
- [IBM's Power 9](https://www.ibm.com/it-infrastructure/power/power9-b)

## Contributing

Contributions are **welcome and very much appreciated**. We are also open to starting research collaborations or mentoring if you are interested in or need assistance implementing new algorithms.
Please refer to [our Contribution Guide](CONTRIBUTING.md) for more details.

## Citing dislib

If you use dislib in a scientific publication, we would appreciate you citing the following paper:

J. Álvarez Cid-Fuentes, S. Solà, P. Álvarez, A. Castro-Ginard, and R. M. Badia, "dislib: Large Scale High Performance Machine Learning in Python," in *Proceedings of the 15th International Conference on eScience*, 2019, pp. 96-105

### Bibtex

```latex
@inproceedings{dislib,
            title       = {{dislib: Large Scale High Performance Machine Learning in Python}},
            author      = {Javier Álvarez Cid-Fuentes and Salvi Solà and Pol Álvarez and Alfred Castro-Ginard and Rosa M. Badia},
            booktitle   = {Proceedings of the 15th International Conference on eScience},
            pages       = {96-105},
            year        = {2019},
 }
```

## Acknowledgements

This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement H2020-MSCA-COFUND-2016-754433.

This work has also received funding from the collaboration project between the Barcelona Supercomputing Center (BSC) and Fujitsu Ltd.

In addition, the development of this software has also been supported by the following institutions:

- Spanish Government under contracts SEV2015-0493, TIN2015-65316 and PID2019-107255G.

- Generalitat de Catalunya under contract 2017-SGR-01414 and the CECH project, co-funded with 50% by the European Regional Development Fund under the
framework of the ERDF Operative Programme for Catalunya 2014-2020.

- European Commission's through the following R&D projects:
  - H2020 I-BiDaaS project (Contract 780787)
  - H2020 BioExcel Center of Excellence (Contracts 823830, and 675728)
  - H2020 EuroHPC Joint Undertaking MEEP Project (Contract 946002)
  - H2020 EuroHPC Joint Undertaking eFlows4HPC Project (Contract 955558)
  - H2020 AI-Sprint project (Contract 101016577)
  - H2020 PerMedCoE  Center of Excellence (Contract 951773)
  - Horizon Europe CAELESTIS project (Contract 101056886)
  - Horizon Europe DT-Geo project (Contract 101058129)
  - Horizon Europe CyclOps project (Contract 101135513)

## License

Apache License Version 2.0, see [LICENSE](LICENSE)
