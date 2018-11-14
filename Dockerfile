FROM compss/compss-ubuntu16:2.4
MAINTAINER COMPSs Support <support-compss@bsc.es>

COPY dislib /dislib
COPY examples /examples
COPY tests /tests
COPY run_tests.sh /
