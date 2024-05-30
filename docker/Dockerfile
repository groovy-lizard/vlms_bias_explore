FROM nvidia/cuda:12.3.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && \
    apt-get update -y && \
    apt-get install -y python3-dev build-essential libssl-dev libffi-dev libjpeg-dev zlib1g-dev libjpeg62 && \
    apt-get install -y wget git ca-certificates curl nginx python3-pip


WORKDIR /build
RUN mkdir -p /opt/ml

COPY requirements.txt .
ENV PATH=/root/.local/bin:$PATH

RUN pip3 install --user --upgrade pip
RUN pip install --user cython
RUN pip3 install --user torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --user -r requirements.txt
