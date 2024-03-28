FROM compss/compss-tutorial:3.3
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib:/opt/COMPSs/Bindings/python/3/:/python-blosc2:/pyeddl
ENV LC_ALL=C.UTF-8
RUN python3 -m pip install flake8 parameterized coverage
RUN git clone https://github.com/Blosc/python-blosc2/ /python-blosc2
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-build.txt
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /python-blosc2/requirements-runtime.txt
RUN git clone https://github.com/deephealthproject/pyeddl.git /pyeddl
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /pyeddl/requirements.txt
RUN cd /python-blosc2 && git submodule update --init --recursive && python3 setup.py build_ext --inplace -- -DDEACTIVATE_AVX2:STRING=ON
RUN python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade -r /dislib/requirements.txt


ENV COMPSS_LOAD_SOURCE false

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
