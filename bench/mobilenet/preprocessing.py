#image 전처리 library
import tensorflow as tf
import numpy as np
from PIL import Image

# 이미지 로드 및 전처리 (for mobilenet)
def run_preprocessing(image_file_path):
    img = Image.open(image_file_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array