#!/bin/bash

# cnn model (mobilenet_v1, mobilenet_v2, inception_v3)
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v1/mobilenet_v1.zip
unzip -q mobilenet_v1.zip && rm mobilenet_v1.zip
mkdir mobilenet_v1/1/ && mv mobilenet_v1/* mobilenet_v1/1/

curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v2/mobilenet_v2.zip
unzip -q mobilenet_v2.zip && rm mobilenet_v2.zip
mkdir mobilenet_v2/1/ && mv mobilenet_v2/* mobilenet_v2/1/

curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/inception_v3/inception_v3.zip
unzip -q inception_v3.zip && rm inception_v3.zip
mkdir inception_v3/1/ && mv inception_v3/* inception_v3/1/

#object detection model (yolo_v5)
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/yolo_v5/yolo_v5.zip
unzip -q yolo_v5.zip && rm yolo_v5.zip
mv yolov5/yolov5s_saved_model yolo_v5 && rm -r yolov5
mkdir yolo_v5/1/ && mv yolo_v5/* yolo_v5/1/

# nlp model (bert_imdb, distilbert_sst2)
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/NLP/bert_imdb.zip
unzip -q bert_imdb.zip && rm bert_imdb.zip
mkdir bert_imdb/1/ && mv bert_imdb/* bert_imdb/1/

crul -O https://edge-inference.s3.us-west-2.amazonaws.com/NLP/distilbert_sst2.zip
unzip -q distilbert_sst2.zip && rm distilbert_sst2.zip
mkdir distilbert_sst2/1/ && mv distilbert_sst2/* distilbert_sst2/1/
