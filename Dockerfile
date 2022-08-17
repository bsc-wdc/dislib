FROM bscwdc/dislib-base:latest
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY . dislib/

ENV PYTHONPATH=$PYTHONPATH:/dislib
ENV LC_ALL=C.UTF-8
RUN python3 -m pip install --upgrade -r /dislib/requirements.txt

ENV COMPSS_LOAD_SOURCE false

RUN sed -i "s/>4</>$(grep -c ^processor /proc/cpuinfo)</g" /opt/COMPSs//Runtime/configuration/xml/resources/default_resources.xml

# Expose SSH port and run SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
