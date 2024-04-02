FROM compss/compss-tutorial:3.3
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib:/opt/COMPSs/Bindings/python/3/:/python-blosc2:/pyeddl
ENV LC_ALL=C.UTF-8
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y libeigen3-dev && \
    apt-get install -y protobuf-compiler && \
    apt-get install -y libprotobuf-dev && \
    apt-get install -y zlib1g-dev && \
    apt-get install -y libgtest-dev
RUN python3 -m pip install flake8 parameterized coverage
RUN git clone https://github.com/Blosc/python-blosc2/ /python-blosc2
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-build.txt
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-runtime.txt
RUN git clone https://github.com/deephealthproject/eddl.git /eddl
RUN cd /eddl && mkdir build && cd build && \
    cmake .. \
    -D BUILD_TARGET=CPU \
    -DBUILD_SUPERBUILD=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CUDA_COMPILER=$(which nvcc) \
    -DCMAKE_PREFIX_PATH=/eddl \
    -DCMAKE_INSTALL_PREFIX=/eddl \
    -DCMAKE_BUILD_TYPE=Release
RUN cd build && \
    make -j$(nproc) && \
    make install && cd .. & cd ..
RUN git clone -b 1.2.0 https://github.com/deephealthproject/pyeddl.git /pyeddl
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /pyeddl/requirements.txt
RUN cd /python-blosc2 && git submodule update --init --recursive && python3 setup.py build_ext --inplace -- -DDEACTIVATE_AVX2:STRING=ON
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements.txt


ENV COMPSS_LOAD_SOURCE false

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
