#image 전처리 library
import numpy as np
import os
import cv2

# 이미지 로드 및 전처리 (for yolo)
def run_preprocessing(image_file_path):
    img = cv2.imread(image_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    return img
