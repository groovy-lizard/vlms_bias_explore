# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.3.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# remove any third-party apt sources to avoid issues with expiring keys
RUN rm -f /etc/apt/sources.list.d/*.list

# install base utils
RUN apt-get update \
    && apt-get install -y build-essential \
    curl \
    ca-certificates \
    sudo \
    git \
    wget \
    bzip2 \
    unzip \
    libsdl1.2-dev \
    libsdl-image1.2-dev \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    yasm \
    cmake \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app


COPY environment.yml /tmp/environment.yml

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init && conda env create -n clipenv --file /tmp/environment.yml
RUN conda init && conda activate clipenv && \
    pip install git+https://github.com/openai/CLIP.git && \
    pip install open_clip_torch


# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

CMD ["/bin/bash"]
