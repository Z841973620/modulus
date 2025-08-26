ARG PYT_VER=22.05
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3

# Specify poetry version
ENV POETRY_VERSION=1.1.13

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Cache dependencies
COPY pyproject.toml ./
COPY poetry.lock ./

# Copy files into container
COPY . /modulus/

# Extract OptiX 7.0.0 SDK and CMake 3.18.2
RUN cd /modulus && ./NVIDIA-OptiX-SDK-7.0.0-linux64.sh --skip-license --include-subdir --prefix=/root
RUN cd /root && \
    wget https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.tar.gz && \
    tar xvfz cmake-3.18.2-Linux-x86_64.tar.gz

# Build libsdf.so
RUN mkdir /modulus/external/pysdf/build && \
    cd /modulus/external/pysdf/build && \
    /root/cmake-3.18.2-Linux-x86_64/bin/cmake .. -DGIT_SUBMODULE=OFF -DOptiX_INSTALL_DIR=/root/NVIDIA-OptiX-SDK-7.0.0-linux64 -DCUDA_CUDA_LIBRARY="" && \
    make -j && \
    mkdir /modulus/external/lib && \
    cp libpysdf.so /modulus/external/lib/

ENV LD_LIBRARY_PATH="/modulus/external/lib:${LD_LIBRARY_PATH}" \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility,video \
    _CUDA_COMPAT_TIMEOUT=90

# Install pysdf
RUN cd /modulus/external/pysdf && python setup.py install

# install tiny-cuda-nn
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/@master#subdirectory=bindings/torch

# Install modulus and dependencies
RUN cd /modulus && poetry config virtualenvs.create false \
    && poetry install --no-interaction

# Copy Pysdf egg file
RUN mkdir /modulus/external/eggs
RUN cp -r /modulus/external/pysdf/dist/pysdf-0.1-py3.8-linux-x86_64.egg /modulus/external/eggs

# Cleanup
RUN rm -rf /root/NVIDIA-OptiX-SDK-7.0.0-linux64 /root/cmake-3.18.2-Linux-x86_64 /modulus/external/pysdf  /modulus/.git*
RUN rm -fv /modulus/setup.py /modulus/setup.cfg /modulus/MANIFEST.in

WORKDIR /examples
