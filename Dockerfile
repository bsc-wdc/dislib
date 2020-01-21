FROM bscwdc/dislib-base:latest
MAINTAINER COMPSs Support <support-compss@bsc.es>

RUN apt-get update -y && apt-get update
RUN apt-get install -y cmake python3-dev libpython3-dev gcc-4.8 libtool python3-numpy python3-pip python3-setuptools
RUN curl -L https://github.com/bsc-dd/hecuba/archive/NumpyWritePartitions.tar.gz | tar -xz
#RUN python3 -m pip install --upgrade pip && python3 -m pip install -r hecuba-NumpyWritePartitions/requirements.txt
WORKDIR hecuba-NumpyWritePartitions
RUN python3 -m pip install -r requirements.txt
RUN python3 setup.py install
WORKDIR /

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
