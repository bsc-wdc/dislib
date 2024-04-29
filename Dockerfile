FROM compss/compss-tutorial:latest
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib:/opt/COMPSs/Bindings/python/3/:/python-blosc2:/usr/lib/python3.8/site-packages/pyeddl-1.3.1-py3.8-linux-x86_64.egg
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/eigen3/Eigen/:/usr/include/eigen3
ENV EDDL_DIR=/pyeddl/third_party/eddl
ENV CPATH="/usr/include/eigen3/:${CPATH}"
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y libeigen3-dev protobuf-compiler libprotobuf-dev zlib1g-dev libgtest-dev && \
    python3 -m pip install flake8 parameterized coverage tensorflow torch && \
    git clone https://github.com/Blosc/python-blosc2/ /python-blosc2 && cd /python-blosc2 && git checkout v2.5.1 && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-build.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-runtime.txt && \
    cd /python-blosc2 && git submodule update --init --recursive && python3 setup.py build_ext --inplace -- -DDEACTIVATE_AVX2:STRING=ON && \
    git clone --recurse-submodules https://github.com/deephealthproject/pyeddl.git /pyeddl && cd /pyeddl && \
    cd third_party/eddl && mkdir build && cd build && cmake .. -D CMAKE_INSTALL_PREFIX=/pyeddl/third_party/eddl -D BUILD_TARGET=CPU -D BUILD_SHARED_LIB=ON -D BUILD_SUPERBUILD=ON -D BUILD_PROTOBUF=ON -D BUILD_TESTS=OFF && \
    make && make install && cd ../.. && \
    python3 -m pip install --upgrade setuptools pip && \
    python3 -m pip install --upgrade numpy 'pybind11<2.6' pytest && cd /pyeddl && python3 setup.py install && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements_gpu.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements_tests.txt

ENV COMPSS_LOAD_SOURCE false

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
