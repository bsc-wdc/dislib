FROM compss/compss-tutorial:3.3
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib:/opt/COMPSs/Bindings/python/3/:/python-blosc2:/pyeddl
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/eigen3/Eigen/
ENV EDDL_DIR=/eddl
ENV CPATH=/usr/include/eigen3/:$CPATH
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y libeigen3-dev protobuf-compiler libprotobuf-dev zlib1g-dev libgtest-dev && \
    python3 -m pip install flake8 parameterized coverage && \
    git clone https://github.com/Blosc/python-blosc2/ /python-blosc2 && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-build.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-runtime.txt && \
    git clone https://github.com/deephealthproject/eddl.git /eddl && \
    cd /eddl && git checkout v1.0.4b && mkdir build && cd build && \
    cmake .. -DBUILD_HPC=OFF -DBUILD_TARGET=CPU -DBUILD_PROTOBUF=OFF && \
    cd /eddl/build && make install && cd ../.. && \
    python3 -m pip install pybind11 && \
    git clone https://github.com/deephealthproject/pyeddl.git /pyeddl && cd /pyeddl && git checkout 1.2.0 \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /pyeddl/requirements.txt && \
    cd /python-blosc2 && git submodule update --init --recursive && python3 setup.py build_ext --inplace -- -DDEACTIVATE_AVX2:STRING=ON
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements.txt


ENV COMPSS_LOAD_SOURCE false

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
