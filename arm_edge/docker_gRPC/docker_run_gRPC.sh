#!/bin/bash

pushd ..
p_pwd=$(pwd)
popd

docker run --rm \
        --gpus all \
        -p 8500:8500 \
        -v $p_pwd/models/:/models/ \
        edge-tf-serving-gRPC:latest