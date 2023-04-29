#!/bin/bash

this_pwd=$(pwd)

docker run --rm \
        --gpus all \
        -p 8500:8500 \
        -v $this_pwd/models/:/models/ \
        edge-tf-serving:latest