<h1 align="center">
  <br>
  <a href="https://www.bsc.es/">
    <img src="docs/bsc_logo.png" alt="Barcelona Supercomputing Center" height="60px">
  </a>
  <br>
  <br>
  The Distributed Computing Library
  <br>
</h1>

<h3 align="center">Distributed Computing library implemented using PyCOMPSs programming model for HPC.</h3>

<p align="center">
  <a href="https://dislib.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/dislib/badge/?version=latest"
         alt="Documentation Status"/>
  </a>
  <a href="https://github.com/bsc-wdc/dislib/releases">
    <img src="https://badge.fury.io/gh/bsc-wdcd%2Fdislib.svg"
         alt="GitHub version">
  </a>
  <a href="https://travis-ci.org/src-d/engine">
    <img src="https://travis-ci.org/bsc-wdc/dislib.svg?branch=master"
         alt="Build Status">
  </a> 
  <a href="https://codecov.io/gh/bsc-wdc/dislib">
    <img src="https://codecov.io/gh/bsc-wdc/dislib/branch/master/graph/badge.svg"
         alt="Code Coverage"/>
  </a>
    
</p>

<p align="center"><b>
    <a href="https://www.bsc.es/research-and-development/software-and-apps/software-list/comp-superscalar/">Website</a> •  
    <a href="https://dislib.readthedocs.io/en/master">Documentation</a> •
</b></p>


## Introduction

The Distributed Computing library is a project which aims to provide distributed machine learning algorithms ready to use as a library. Is is developed on top of PyCOMPSs programming model and is being developed by the [Workflows and Distributed Computing group](https://github.com/bsc-wdc) of the [Barcelona Supercomputing Center](https://www.bsc.es/). The library is designed to allow easy local development through docker. Once the code is finished, it can be run directly into a supercomputer / cloud without any further changes.

Currently the following Supercomputers have already PyCOMPSs installed and ready to use. If you need help configuring your own cluster or supercomputer drop us an email and we will be pleased to help.

- Marenostrum 4 - Barcelona Supercomputing Center (BSC)
- Minotauro - Barcelona Supercomputing Center (BSC)
- Nord 3 - Barcelona Supercomputing Center (BSC)
- Cobi - Barcelona Supercomputing Center (BSC)
- Juron - Jülich Supercomputing Centre (JSC)
- Jureca - Jülich Supercomputing Centre (JSC)
- Ultraviolet - The Genome Analysis Center (TGAC)
- Archer - University of Edinburgh’s Advanced Computing Facility (ACF)

Aso supported:
- [Intel SSF architectures](https://www.intel.com/content/www/us/en/high-performance-computing/ssf-architecture-specification.html)
- [IBM's Power 9](https://www.ibm.com/it-infrastructure/power/power9-b).

## Contents

- [Quickstart](#quickstart)
- [Contributing](#contributing)
- [License](#license)


## Quickstart

Folow the steps below to get started with wdc Dislib.

#### 1. Install Docker

Follow these instructions:

- [Docker for Mac](https://store.docker.com/editions/community/docker-ce-desktop-mac). Or, if you prefer to use [Homebrew](https://brew.sh/):
  ```bash
  brew cask install docker
  ```
- [Docker for Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1)
  ```bash
  sudo apt-get update
  sudo apt-get install docker-ce
  ```
- [Docker for Arch Linux](https://wiki.archlinux.org/index.php/Docker#Installation)
  ```bash
  sudo pacman -S docker
  ```


#### 2. Install the dislib

Download the **[latest release](https://github.com/bsc-wdc/dislib/releases)** for MacOS (Darwin) or Linux.


Extract the tar file from your terminal:
```bash
tar -zxvf dislib_v0.1.0.tar.gz
```

Move it into your local bin folder to be executable from anywhere:
```bash
sudo mv dislib/* /usr/local/bin/
```



#### 3. Start dislib in your development directory

Initialize the dislib where your source code will be (you can reinit anytime). This will allow docker to access your local code and run it inside the container.

```
# Without a path it operates on the current working directory.
dislib init

# You can also provide a path
dislib init /home/user/replace/path/
```

#### 4. Run a dislib example application

First clone dislib repo:

```bash
git clone https://github.com/bsc-wdc/dislib.git
```

Init the dislib environment in the examples folder. This will mount the examples directory inside the container. The exec the desired example:

```bash
cd dislib/examples
dislib init
dislib exec clustering_comparison.py
```

The source files path are resolved from the init directory. Notice the difference if the dislib is initialized in the root of the repo:

```bash
cd dislib
dislib init
dislib exec examples/clustering_comparison.py
```

The log files of the execution can be found at $HOME/.COMPSs.


## Contributing

Contributions are **welcome and very much appreciated**. We are also open to starting research collaborations or mentoring if you are interested in or need assistance to implement new algorithms.
Please refer [to our Contribution Guide](CONTRIBUTING.md) for more details.


## License

GNU GENERAL PUBLIC LICENSE Version 3.0, see [LICENSE](LICENSE)
