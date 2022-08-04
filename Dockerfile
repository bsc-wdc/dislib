FROM bscwdc/dislib-base:latest
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib
ENV LC_ALL=C.UTF-8
RUN pip3 install --upgrade -r /dislib/requirements.txt

ENV COMPSS_LOAD_SOURCE false

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
