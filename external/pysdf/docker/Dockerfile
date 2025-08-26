FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        g++ \
        make \
        wget  && \  
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -nc --no-check-certificate -P /var/tmp https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.sh && \
    mkdir -p /usr/local && \
    /bin/sh /var/tmp/cmake-3.20.0-linux-x86_64.sh --prefix=/usr/local --skip-license
ENV PATH=/usr/local/bin:$PATH
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 \
      python3-pip
RUN pip3 install numpy
COPY optix7.2/SDK /usr/local/optix/SDK
COPY optix7.2/include /usr/local/optix/include
