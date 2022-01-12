runcompss \
  --project=projectAgents4cpu.xml \
  --lang=python \
  -d \
  -t \
  --resources=resources.xml \
  --pythonpath=$(pwd)/src:/home/bscuser/git/dislib/tests/performance/mn4/scripts:/home/bscuser/git/dislib/tests/performance/mn4/tests:/home/bscuser/git/dislib/ \
  --python_interpreter=python3 \
  gridSearchNested.py 2 2
