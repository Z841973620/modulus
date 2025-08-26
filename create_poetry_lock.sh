#!/bin/sh

TMP_CONTAINER_IMG="modulus-poetry"

# Build temporary container image to create new lock file
docker build -t ${TMP_CONTAINER_IMG} -f Dockerfile.poetry .

# Copy lock file out of temp container and change permissions
docker run --rm -iv${PWD}:/host-volume ${TMP_CONTAINER_IMG} sh -s <<EOF
cd /modulus
chown -v $(id -u):$(id -g) poetry.lock
cp -va poetry.lock /host-volume
EOF
