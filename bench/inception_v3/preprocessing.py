#image 전처리 library
import numpy as np
from PIL import Image
import os

# 이미지 로드 및 전처리 (for inception)
def run_preprocessing(image_file_path):
    img = Image.open(image_file_path)
    img = img.resize((299, 299))
    img_array = np.array(img)
    img_array = (img_array - np.mean(img_array)) / np.std(img_array) 
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array