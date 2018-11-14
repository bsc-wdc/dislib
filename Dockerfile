FROM compss/compss-ubuntu16:2.4
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY dislib /home/dislib/dislib
COPY examples /home/dislib/examples
COPY tests /home/dislib/tests
COPY run_tests.sh /home/dislib/
