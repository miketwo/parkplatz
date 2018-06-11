FROM tensorflow/tensorflow:latest-gpu

RUN rm /etc/apt/sources.list.d/*
ADD requirements.txt /tmp
RUN apt-get update && apt-get install -y python-setuptools python-dev build-essential python3 python3-pip
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /srv
