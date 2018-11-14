FROM compss/compss-ubuntu16:2.4

MAINTAINER COMPSs Support <support-compss@bsc.es>


RUN pip3 install codecov coverage && \
# Enable ssh to localhost
    yes yes | ssh-keygen -f /root/.ssh/id_rsa -t rsa -N '' > /dev/null && \
    cat /root/.ssh/id_rsa.pub > /root/.ssh/authorized_keys 

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

