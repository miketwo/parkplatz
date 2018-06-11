#!/bin/bash

echo "Go to /srv and run working.py"
xhost + local:docker
docker run --rm -it \
    --net=host \
    -v $PWD:/srv \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /root/.Xauthority:/root/.Xauthority:rw \
    valian/docker-python-opencv-ffmpeg bash
