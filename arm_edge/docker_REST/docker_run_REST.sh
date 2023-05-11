#!/bin/bash

pushd ..
p_pwd=$(pwd)
popd

docker run --rm \
        --gpus all \
        -p 8501:8501 \
        -v $p_pwd/models/:/models/ \
        edge-tf-serving-REST:latest