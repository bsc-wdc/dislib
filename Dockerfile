FROM compss/compss-tutorial:3.3
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY . dislib/

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y libeigen3-dev protobuf-compiler libprotobuf-dev zlib1g-dev libgtest-dev

ENV PYTHONPATH=$PYTHONPATH:/dislib:/opt/COMPSs/Bindings/python/3/:/python-blosc2
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/eigen3/Eigen/
ENV EDDL_DIR=/eddl
ENV CPATH="/usr/include/eigen3/:${CPATH}"
RUN python3 -m pip install flake8 parameterized coverage
RUN git clone https://github.com/Blosc/python-blosc2/ /python-blosc2
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-build.txt
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-runtime.txt
RUN cd /python-blosc2 && git submodule update --init --recursive && python3 setup.py build_ext --inplace -- -DDEACTIVATE_AVX2:STRING=ON
RUN git clone --recurse-submodules https://github.com/deephealthproject/pyeddl.git /pyeddl && cd /pyeddl && \
    cd third_party/eddl && mkdir build && cd build && cmake .. -D BUILD_SHARED_LIB=ON -D BUILD_PROTOBUF=ON -D BUILD_TESTS=OFF -D BUILD_SUPERBUILD=ON && \
    make -j$(nproc) && make install && cd ../.. && \
    python3 -m pip install --upgrade numpy 'pybind11<2.6' pytest && cd /pyeddl && python3 setup.py install
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements.txt


ENV COMPSS_LOAD_SOURCE false

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]