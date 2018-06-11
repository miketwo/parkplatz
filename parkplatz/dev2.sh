#!/bin/bash -e

echo "Go to /srv and run working.py"
xhost + local:docker

echo "NVidea devices"
ls -la /dev | grep nvidia


nvidia-docker build -t training .
nvidia-docker run --rm -it \
    --runtime=nvidia \
    -v $PWD:/srv \
    training bash


