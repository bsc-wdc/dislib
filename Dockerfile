FROM phusion/baseimage:0.11
#FROM python:3.6.7-slim
#FROM openjdk:8-slim 

MAINTAINER COMPSs Support <support-compss@bsc.es>

# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

RUN apt-get update && apt-get install -y \
    openjdk-8-jre-headless \
    python3-minimal \
    python3-pip \
    openssh-server 
    uuid-runtime && \
# Coverage stuff 
    pip3 install codecov coverage && \
# Default to python3
    ln -s /usr/bin/python3 /usr/bin/python && \
# Clean up APT.
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
# Enable ssh to localhost
    yes yes | ssh-keygen -f /root/.ssh/id_rsa -t rsa -N '' > /dev/null && \
    cat /root/.ssh/id_rsa.pub > /root/.ssh/authorized_keys 

# Copy 2.4 COMPSs installation
COPY --from=compss/compss-ubuntu16:2.4 /opt/COMPSs /opt/COMPSs
#COPY --from=compss/compss-ubuntu16:2.4 /etc/profile.d/compss.sh /etc/profile.d/compss.sh
COPY dislib dislib
COPY examples examples
COPY tests tests
COPY run_tests.sh /
COPY run_coverage.sh /

ENV PATH=$PATH:/opt/COMPSs/Runtime/scripts/user:/opt/COMPSs/Bindings/c/bin
ENV CLASSPATH=$CLASSPATH:/opt/COMPSs/Runtime/compss-engine.jar
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
#ENV MPI_HOME=/usr/lib64/openmpi
#ENV LD_LIBRARY_PATH=/usr/lib64/openmpi/lib

