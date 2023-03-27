import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#gRPC Import
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import concurrent.futures

import time
import numpy as np
from PIL import Image


# 병렬 처리할 작업 횟수 지정
num_tasks = 30
# 모델 서버의 주소 및 포트 번호 설정
server_address = 'localhost:8500'
model_name = "mobilenet_v1"
image_file_path = "../../../dataset/imagenet/imagenet_1000_raw/n01843383_1.JPEG"

# 이미지 로드 및 전처리
img = Image.open(image_file_path)
img = img.resize((224, 224))
img_array = np.array(img)
img_array = img_array.astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)

# gRPC 채널 생성
channel = grpc.insecure_channel(server_address,options=[('grpc.max_send_message_length', 50 * 1024 * 1024), ('grpc.max_recieve_message_length', 50 * 1024 * 1024)])
# 만약 HTTPS 사용 시, 아래와 같이 설정합니다.
#channel = grpc.secure_channel(server_address,grpc.ssl_channel_credentials(),options=[('grpc.max_send_message_length', 50 * 1024 * 1024), ('grpc.max_recieve_message_length', 50 * 1024 * 1024)])

# gRPC 스텁 생성
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# gRPC 요청 생성
data = tf.make_tensor_proto(img_array)
request = predict_pb2.PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = 'serving_default'
request.inputs['input_1'].CopyFrom(data)

# 병렬 처리할 함수 정의
def predict():
    start_time = time.time()
    result = stub.Predict(request, timeout=100.0)
    end_time = time.time()
    return result, end_time - start_time

start_time = time.time()
# gRPC 요청 병렬 처리
with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
    futures = [executor.submit(lambda: predict()) for _ in range(num_tasks)]

# 결과 출력
results = []
for future in concurrent.futures.as_completed(futures):
    result, thread_time  = future.result()
    results.append(result.outputs['dense_1'].float_val)
    #print(result.outputs['dense_1'].float_val)
    print("Thread execution time: ", thread_time)

end_time = time.time()
print(end_time - start_time)

