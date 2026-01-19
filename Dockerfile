FROM compss/compss:3.3.3
LABEL maintainer="COMPSs Support <support-compss@bsc.es>"

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib:/opt/COMPSs/Bindings/python/3/:/python-blosc2
ENV LC_ALL=C.UTF-8
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y --no-install-recommends libeigen3-dev protobuf-compiler libprotobuf-dev zlib1g-dev libgtest-dev git curl && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install tensorflow torch && \
    git clone https://github.com/Blosc/python-blosc2/ /python-blosc2 && cd /python-blosc2 && git checkout v2.5.1 && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-build.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-runtime.txt && \
    git submodule update --init --recursive && python3 setup.py build_ext --inplace -- -DDEACTIVATE_AVX2:STRING=ON && \
    python3 -m pip install --upgrade setuptools pip && \
    python3 -m pip install --upgrade 'pybind11<2.6' && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements_tests.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements_ci.txt

ENV COMPSS_LOAD_SOURCE=false

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
