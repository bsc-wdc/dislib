FROM ubuntu:20.04
MAINTAINER COMPSs Support <support-compss@bsc.es>

# =============================================================================
# Configuration required to use the image for jenkins testing
# =============================================================================

ENV DEBIAN_FRONTEND noninteractive
ENV COMPSS_LOAD_SOURCE false

RUN apt-get update && \
# Install Packages
    apt-get install -y --no-install-recommends \
    git  \
    vim \
    wget \
    openssh-server \
    sudo && \
# Create Jenkins User
    useradd jenkins -m -s /bin/bash && \
# Add the jenkins user to sudoers
    echo "jenkins  ALL=(ALL)  NOPASSWD:ALL" >> /etc/sudoers && \
# Enable ssh to localhost for user root & jenkins
    yes yes | ssh-keygen -f /root/.ssh/id_rsa -t rsa -N '' > /dev/null && \
    cat /root/.ssh/id_rsa.pub > /root/.ssh/authorized_keys && \
    cp -r /root/.ssh /home/jenkins && \
# Make sure jenkins owns his files
    chown -R jenkins /home/jenkins/ && \
    chgrp -R jenkins /home/jenkins/ && \
# Enable repo compression
    git config --global core.compression 9 && \
# =============================================================================
# Dependencies for building COMPSs
# =============================================================================
# Build dependencies
    apt-get install -y --no-install-recommends maven \
# Runtime dependencies
    openjdk-8-jdk graphviz xdg-utils \
# Bindings-common-dependencies
    libtool automake build-essential \
# C-binding dependencies
    libboost-all-dev libxml2-dev csh \
# Extrae dependencies
    libxml2 gfortran libpapi-dev papi-tools \
# Misc. dependencies
    openmpi-bin openmpi-doc libopenmpi-dev uuid-runtime curl bc cmake && \
# Python-binding and dislib dependencies
    apt-get install -y python3 python3-pip python3-dev graphviz-dev && \
    python3 -m pip install coverage ipython flake8 codecov parameterized cvxpy==1.1.5 && \
    python3 -m pip install pycodestyle pydocstyle  mpi4py  numpy dill guppy3 memory_profiler matplotlib decorator \
            jupyter pytest nbval pytest-cov pytest-notebook ipyparallel jupyter_nbextensions_configurator jupyterlab \ 
            pytest nbval pytest-cov pytest-html-profiling pytest-metadata pytest-profiling pytest-subprocess pytest-sugar spacy  && \
    python3 -m ipykernel install && \
    python3 -m spacy download en_core_web_sm && \
# Configure user environment
# =============================================================================
# System configuration
# =============================================================================
# Add environment variables
    echo "JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" >> /etc/environment && \
    echo "MPI_HOME=/usr/lib/openmpi" >> /etc/environment && \
    echo "LD_LIBRARY_PATH=/usr/lib/openmpi/lib" >> /etc/environment && \
    mkdir /run/sshd && \
# Clone framework files for installation
    git clone --depth=1 --branch 3.0 https://github.com/bsc-wdc/compss.git framework && \
# Install COMPSs
    cd /framework && \
    ./submodules_get.sh && \
    sudo -E /framework/builders/buildlocal -NpKT /opt/COMPSs && \
    rm -rf /framework /root/.m2 /root/.cache /home/jenkins/.COMPSs /tmp/* && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
ENV PATH $PATH:/opt/COMPSs/Runtime/scripts/user:/opt/COMPSs/Bindings/c/bin:/opt/COMPSs/Runtime/scripts/utils
ENV CLASSPATH $CLASSPATH:/opt/COMPSs/Runtime/compss-engine.jar
ENV LD_LIBRARY_PATH /opt/COMPSs/Bindings/bindings-common/lib:$JAVA_HOME/jre/lib/amd64/server
