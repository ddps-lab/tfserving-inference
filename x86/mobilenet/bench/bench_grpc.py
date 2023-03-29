#import grpc module
import sys
sys.path.append('../../../bench')
import grpc_module
from tensorflow_serving.apis import predict_pb2
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
#google sheet api 관련 library
import gspread_module


# 병렬 처리할 작업 횟수 지정
num_tasks = 30

# 모델 서버의 주소 및 포트 번호, 모델 이름 설정
server_address = 'localhost:8500'
#https 사용시 1로 설정, http 사용시 0으로 설정
usehttps = 1
model_name = "mobilenet_v1"
image_file_path = "../../../dataset/imagenet/imagenet_1000_raw/n01843383_1.JPEG"

stub = grpc_module.create_grpc_stub(server_address, usehttps)

# 이미지 로드 및 전처리 (for mobilenet)
img = Image.open(image_file_path)
img = img.resize((224, 224))
img_array = np.array(img)
img_array = img_array.astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)

# gRPC 요청 생성
data = tf.make_tensor_proto(img_array)
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

all_inference_start_time = time.time()
# gRPC 요청 병렬 처리
with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
    futures = [executor.submit(lambda: predict()) for _ in range(num_tasks)]

inference_times_include_network_latency = np.array()
# 결과 출력
for future in concurrent.futures.as_completed(futures):
    result, thread_elapsed_time  = future.result()
    # results.append(result.outputs['dense_1'].float_val)
    np.append(inference_times_include_network_latency, thread_elapsed_time)
    print("Thread execution time: ", thread_elapsed_time)

all_inference_end_time = time.time()
all_inference_elapsed_time = all_inference_end_time - all_inference_start_time
print(inference_times_include_network_latency)
print("all_elapsed_time",all_inference_elapsed_time)
print("maxtime:",np.max(inference_times_include_network_latency))
print("mintime:",np.min(inference_times_include_network_latency))
print("avgtime:",np.avg(inference_times_include_network_latency))

