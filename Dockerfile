FROM compss/compss-tutorial:latest
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib:/opt/COMPSs/Bindings/python/3/:/python-blosc2
ENV LC_ALL=C.UTF-8
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y libeigen3-dev protobuf-compiler libprotobuf-dev zlib1g-dev libgtest-dev && \
    python3 -m pip install flake8 parameterized coverage tensorflow torch && \
    git clone https://github.com/Blosc/python-blosc2/ /python-blosc2 && cd /python-blosc2 && git checkout v2.5.1 && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-build.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-runtime.txt && \
    cd /python-blosc2 && git submodule update --init --recursive && python3 setup.py build_ext --inplace -- -DDEACTIVATE_AVX2:STRING=ON && \
    python3 -m pip install --upgrade setuptools pip && \
    python3 -m pip install --upgrade numpy 'pybind11<2.6' pytest && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements.txt

ENV COMPSS_LOAD_SOURCE false

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
