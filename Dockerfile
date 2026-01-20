FROM compss/compss:3.3.3
LABEL maintainer="COMPSs Support <support-compss@bsc.es>"

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib:/opt/COMPSs/Bindings/python/3/
ENV LC_ALL=C.UTF-8

# NOTE: remove /root/.cache after pip install (in the same layer) to save image size
# NOTE: --trusted-host is required in some environments due to TLS interception / missing CA certs in base image
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y --no-install-recommends libeigen3-dev protobuf-compiler libprotobuf-dev zlib1g-dev libgtest-dev git curl && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip setuptools torch 'pybind11<2.6' && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements_tests.txt && \
    python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements_ci.txt && \
    rm -rf /root/.cache
    
ENV COMPSS_LOAD_SOURCE=false

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
