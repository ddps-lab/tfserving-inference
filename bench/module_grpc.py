import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#gRPC Import
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

def create_grpc_stub(server_address, use_https):
  # gRPC 채널 생성
  if (use_https):
    channel = grpc.secure_channel(server_address,grpc.ssl_channel_credentials(),options=[('grpc.max_send_message_length', 50 * 1024 * 1024), ('grpc.max_recieve_message_length', 50 * 1024 * 1024)])
  else:
    channel = grpc.insecure_channel(server_address,options=[('grpc.max_send_message_length', 50 * 1024 * 1024), ('grpc.max_recieve_message_length', 50 * 1024 * 1024)])

  # gRPC 스텁 생성
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  return stub