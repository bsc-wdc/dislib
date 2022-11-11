## Quickstart guide

There are two ways in which you can get started with dislib. You can perform 
a [manual installation](#manual-installation), or you can download our 
ready-to-use [docker image](#using-docker). 

### Manual installation

#### Dependencies

dislib currently requires:

* PyCOMPSs >= 2.8
* Scikit-learn >= 1.1.1
* Scipy >= 1.3.0
* NumPy >= 1.18.1
* cvxpy >= 1.1.5

Some of the examples also require matplotlib >= 2.2.3 and pandas >= 0.24.2. 
numpydoc >= 0.8.0 is requried to build the documentation.

#### Installation steps 

1. Check which PyCOMPSs version to install. 
    * Latest dislib release requires **PyCOMPSs 2.8** or greater (check [here](https://github.com/bsc-wdc/dislib/releases) for information about other releases).
    
2. Install PyCOMPSs following these [instructions](https://compss-doc.readthedocs.io/en/stable/Sections/01_Installation.html).

3. Install the latest dislib version with ``pip3 install dislib``.
   * **IMPORTANT:** dislib requires the ``pycompss`` Python module. However, this command will **NOT** install the module automatically. The module should be available after manually installing PyCOMPSs following the instructions in step 2. For more information on this, see [here](https://github.com/bsc-wdc/dislib/issues/190).

4. You can check that everything works fine by running one of our examples:

    * Download the latest source code [here](https://github.com/bsc-wdc/dislib/releases/latest).
    
    * Extract the contents of the tar package.
    
    ```bash
    tar xzvf dislib-X.Y.Z.tar.gz
    ```
    
    * Run an example application.
    
    ```bash
    runcompss --python_interpreter=python3 dislib-X.Y.Z/examples/kmeans.py    
    ```

### Using docker

#### 1. Install Docker and docker-py

**Warning:** requires docker version >= 17.12.0-ce


1. Follow these instructions

   - [Docker for Mac](https://store.docker.com/editions/community/docker-ce-desktop-mac). Or, if you prefer to use [Homebrew](https://brew.sh/).

   - [Docker for Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1).

   - [Docker for Arch Linux](https://wiki.archlinux.org/index.php/Docker#Installation).

Be aware that for some distros the docker package has been renamed from `docker` to `docker-ce`. Make sure you install the new package.

2. Add user to docker group to run dislib as a non-root user.

    - [Instructions](https://docs.docker.com/install/linux/linux-postinstall/)


3. Check that docker is correctly installed

```
docker --version
docker ps # this should be empty as no docker processes are yet running.
```

4. Install [docker-py](https://docker-py.readthedocs.io/en/stable/)

```
pip3 install docker
```

#### 2. Install dislib

```
pip3 install dislib
```

This should add the dislib executable to your path.

#### 3. Start dislib in your development directory

Initialize dislib where your source code will be (you can re-init anytime). This will allow docker to access your local code and run it inside the container.

**Note** that the first time dislib needs to download the docker image from the registry, and it may take a while.
```
# Without a path it operates on the current working directory.
dislib init

# You can also provide a path
dislib init /home/user/replace/path/
```

#### 4. Running applications

**Note**: running the docker dislib does not work with applications with GUI or with visual plots such as `examples/clustering_comparison.py`).

First clone dislib repo and checkout release branch vX.Y.Z (docker version and dislib code should preferably be the same to avoid inconsistencies):

```bash
git clone https://github.com/bsc-wdc/dislib.git
```

Init the dislib environment in the root of the repo.
The source files path are resolved from the init directory which sometimes can be confusing. As a rule of thumb, initialize the library in a current directory and check the paths are correct running the file with `python3 path_to/file.py` (in this case `python3 examples/rf_iris.py`).

```bash
cd dislib
dislib init
dislib exec examples/rf_iris.py
```

The log files of the execution can be found at $HOME/.COMPSs.

You can also init the library inside the examples folder. This will mount the examples directory inside the container so you can execute it without adding the path:
```bash
cd dislib/examples
dislib init
dislib exec rf_iris.py
```

#### 5. Running Jupyter notebooks

Notebooks can be run using the `dislib jupyter` command. Run the
following snippet from the root of the project:

```bash
dislib init
dislib jupyter ./notebooks
```

An alternative and more flexible way of starting jupyter is using the
`dislib run` command in the following way:

```bash
dislib run jupyter-notebook ./notebooks --ip=0.0.0.0  --allow-root
```

Access your notebook by ctrl-clicking or copy pasting into the browser the link shown on the CLI (e.g. http://127.0.0.1:8888/?token=TOKEN_VALUE).

If the notebook process is not properly closed, you might get the following warning when trying to start jupyter notebooks again:

`The port 8888 is already in use, trying another port.`

To fix it, just restart the dislib container with `dislib init`.
#### 6. Adding more nodes


**Note**: adding more nodes is still in beta phase. Please report issues, suggestions, or feature requests on [Github](https://github.com/bsc-wdc/dislib).

To add more computing nodes, you can either let docker create more workers for you or manually create and config a custom node.

For docker just issue the desired number of workers to be added. For example, to add 2 docker workers:
```
dislib components add worker 2
```

You can check that both new computing nodes are up with:

```
dislib components list
```

If you want to add a custom node it needs to be reachable through ssh without user. Moreover, dislib will try to copy the `working_dir` there, so it needs write permissions for the scp.

For example, to add the local machine as a worker node:

```
dislib components add worker '127.0.0.1:6'
```

* '127.0.0.1': is the IP used for ssh (can also be a hostname like 'localhost' as long as it can be resolved).
* '6': desired number of available computing units for the new node.

**Please be aware** that `dislib components` will not list your custom nodes because they are not docker processes and thus it can't be verified if they are up and running.
