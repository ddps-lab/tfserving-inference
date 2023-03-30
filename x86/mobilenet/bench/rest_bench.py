#image 전처리 library
import numpy as np
from PIL import Image

#시간 측정 library
import time

#REST 요청 관련 library
import requests
import json

#병렬처리 library
import concurrent.futures

#sheet 데이터 입력 모듈
import sys
sys.path.append('../../../bench')
import put_data_into_sheet
import variables

# 병렬 처리할 작업 횟수 지정
num_tasks = 30

# 저장할 google spread sheet id
spreadsheet_id = variables.spreadsheet_id
server_address = variables.rest_server_address
model_name = "mobilenet_v1"
image_file_path = "../../../dataset/imagenet/imagenet_1000_raw/n01843383_1.JPEG"

def send_request(data):
    headers = {"content-type": "application/json"}
    url = server_address + model_name + ":predict"
    request_time = time.time()
    response = requests.post(url, data=data, headers=headers)
    response_time = time.time()
    elapsed_time = response_time - request_time
    result = json.loads(response.text)
    return result, elapsed_time

def img_preprocessing(image_file_path):
    img = Image.open(image_file_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    data = json.dumps({"instances": img_array.tolist()})
    return data

data = img_preprocessing(image_file_path)

# REST 요청 병렬 처리
with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
    futures = [executor.submit(send_request, data) for _ in range(num_tasks)]

inference_times_include_network_latency = []
for future in concurrent.futures.as_completed(futures):
    result, thread_elapsed_time = future.result()
    inference_times_include_network_latency.append(thread_elapsed_time)

put_data_into_sheet.put_data(spreadsheet_id, inference_times_include_network_latency, num_tasks)

