# Quickstart guide

There are two ways in which you can get started with dislib. You can perform
a [manual installation](#manual-installation), or you can download our
ready-to-use [docker image](#using-docker).

## Manual installation

### Dependencies

dislib currently requires:

* PyCOMPSs >= 3.2
* scikit-learn >= 1.7
* scipy >= 1.13
* numpy >= 2.0
* cvxpy >= 1.4.2
* cbor2 >= 5.4.0

Some of the examples also require matplotlib and pandas.
numpydoc is required to build the documentation.
GPU-accelerated algorithms require both [PyTorch](https://pytorch.org/) and [CuPy](https://cupy.dev/). CuPy must match your CUDA version, so install the right variant for your system (e.g. `pip install cupy-cuda12x` for CUDA 12). Check the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for details.

### Installation steps

1. Check which PyCOMPSs version to install.
    * Latest dislib release requires **PyCOMPSs 3.2** or greater (check [the releases page](https://github.com/bsc-wdc/dislib/releases) for information about other releases).

2. Install PyCOMPSs following these [installation instructions](https://compss-doc.readthedocs.io/en/stable/Sections/01_Installation_Configuration/02_Installation.html).

3. Install the latest dislib version with ``pip3 install dislib``.
   * **IMPORTANT:** dislib requires the ``pycompss`` Python module. However, this command will **NOT** install the module automatically. The module should be available after manually installing PyCOMPSs following the instructions in step 2. For more information on this, see [issue #190](https://github.com/bsc-wdc/dislib/issues/190).

4. You can check that everything works fine by running one of our examples:

    * Download the latest source code from the [latest release](https://github.com/bsc-wdc/dislib/releases/latest).

    * Extract the contents of the tar package.

    ```bash
    tar xzvf dislib-X.Y.Z.tar.gz
    ```

    * Run an example application.

    ```bash
    runcompss --python_interpreter=python3 dislib-X.Y.Z/examples/kmeans.py
    ```

## Using docker

### 1. Install Docker

1. Follow these instructions:

    * [Docker for Mac](https://store.docker.com/editions/community/docker-ce-desktop-mac). Or, if you prefer to use [Homebrew](https://brew.sh/).
    * [Docker for Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1).
    * [Docker for Arch Linux](https://wiki.archlinux.org/index.php/Docker#Installation).

    Be aware that the docker package has been renamed from `docker` to `docker-ce` for some distributions.
    Make sure you install the new package.

2. Add the user to the docker group to run dislib as a non-root user.

    * [Instructions](https://docs.docker.com/install/linux/linux-postinstall/)

3. Check that docker is correctly installed.

    ```bash
    docker --version
    docker ps # this should be empty as no docker processes are yet running.
    ```

### 2. Pull the image

```bash
docker pull bscwdc/dislib
```

If you need PyTorch support, use the torch variant instead:

```bash
docker pull bscwdc/dislib:torch
```

### 3. Running applications

Start an interactive container:

```bash
docker run -it --rm -d --name dislib bscwdc/dislib
docker exec -it dislib bash
```

Inside the container, install any specific dependency required by the application you want to run, for instance:

```bash
pip install matplotlib pandas
```

Run any example using `runcompss`:

```bash
runcompss --python_interpreter=python3 /dislib/examples/rf_iris.py
```

The log files of the execution can be found at `$HOME/.COMPSs` inside the container.

### 4. Running Jupyter notebooks

Clone the COMPSs tutorial apps repository to access more notebooks:

```bash
git clone https://github.com/bsc-wdc/tutorial_apps.git
```

Start a container with port 8888 exposed and your notebooks directory mounted:

```bash
docker run -it --rm -p 8888:8888 --name dislib -v "$(pwd)/tutorial_apps":/tutorial_apps bscwdc/dislib
docker exec -it dislib bash
```

Install Jupyter with pip inside the container and start Jupyter:

```bash
pip install jupyter tabulate matplotlib
jupyter-notebook --ip=0.0.0.0 --allow-root /tutorial_apps/python/notebooks/syntax/
```

Access your notebook by ctrl-clicking or copy-pasting into the browser the link shown in the terminal (e.g. `http://127.0.0.1:8888/?token=TOKEN_VALUE`).


Finally, choose a notebook to test dislib. For instance:

```
9_Dislib_demo.ipynb
```
