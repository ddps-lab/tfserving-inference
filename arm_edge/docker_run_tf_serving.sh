#!/bin/bash

this_pwd=$(pwd)

docker run --rm \
        --gpus all \
        -p 8500:8500 \
        -p 8501:8501 \
        -v $this_pwd/models/:/models/ \
        edge-tf-serving:latest