#import grpc module
import sys
sys.path.append('../../../bench')
import module_grpc
from tensorflow_serving.apis import predict_pb2
import variables

#tf log setting
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#image 전처리 library
import tensorflow as tf
import numpy as np
from PIL import Image

#병렬처리 library
import concurrent.futures

#시간 측정 library
import time

#sheet 데이터 입력 모듈
import put_data_into_sheet


# 병렬 처리할 작업 횟수 지정
num_tasks = 30


spreadsheet_id = variables.spreadsheet_id
server_address = variables.grpc_server_address
usehttps = variables.usehttps
model_name = "mobilenet_v1"
image_file_path = "../../../dataset/imagenet/imagenet_1000_raw/n01843383_1.JPEG"

stub = module_grpc.create_grpc_stub(server_address, usehttps)

# 이미지 로드 및 전처리 (for mobilenet)
def img_preprocessing(image_file_path):
    img = Image.open(image_file_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    data = tf.make_tensor_proto(img_array)
    return data

data = img_preprocessing(image_file_path)

# gRPC 요청 생성
request = predict_pb2.PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = 'serving_default'
request.inputs['input_1'].CopyFrom(data)

# 병렬 처리할 함수 정의
def predict():
    request_time = time.time()
    result = stub.Predict(request, timeout=100.0)
    response_time = time.time()
    elapsed_time = response_time - request_time
    return result, elapsed_time

# gRPC 요청 병렬 처리
with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
    futures = [executor.submit(lambda: predict()) for _ in range(num_tasks)]

inference_times_include_network_latency = []
# 결과 출력
for future in concurrent.futures.as_completed(futures):
    result, thread_elapsed_time  = future.result()
    inference_times_include_network_latency.append(thread_elapsed_time)

put_data_into_sheet.put_data(spreadsheet_id, inference_times_include_network_latency, num_tasks)