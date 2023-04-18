#!/bin/bash

docker run --rm \
        --gpus all \
        -p 8500:8500 \
        -v ./models/:/models/ \
        edge-tf-serving:latest