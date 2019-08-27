#!/bin/bash -e

docker build --no-cache -t bscwdc/dislib-base:latest .

docker login -u ${dh_username} -p ${dh_password}
docker push bscwdc/dislib-base:latest
