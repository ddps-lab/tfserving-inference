# from module import put_data_into_sheet
# put_data_into_sheet.put_data(variables.rest_spreadsheet_id, result, variables.num_tasks)

import argparse
import roundrobin
import time
from threading import Thread
import importlib
import inference_request_workload_manager


parser = argparse.ArgumentParser()
parser.add_argument('--edge', default=None, type=str)

args = parser.parse_args()
edges_to_inference = args.edge

# grpc or rest
request_type = 'rest'
# workload file name
workload_file_name = 'inference_request_workload.pickle'
# 서버의 포트 정보
grpc_port = 8500
rest_port = 8501

# 각 장비의 ip와 로드된 모델들을 설정해주어야함.
edges_info = {'nvidia-xavier2': {'ip_addr': 'nvidia-xavier2',
                                 'models': ['mobilenet_v1', 'mobilenet_v2', 'inception_v3', 'yolo_v5']
                                 },
              'nvidia-tx2': {'ip_addr': 'nvidia-tx2',
                             'models': ['mobilenet_v1', 'mobilenet_v2', 'inception_v3', 'yolo_v5']
                             },
              'nvidia-nano1': {'ip_addr': 'nvidia-nano1',
                               'models': ['mobilenet_v1']
                               }
              }


# --edge 옵션이 없을 시 등록되어 있는 모든 장비들에 추론 요청, 요청장비들은 edges_info에 등록되어 있어야함. 입력 형식은 'a, b, ...'
registered_edges = list(edges_info.keys())

if request_type not in ['grpc', 'rest']:
    print(f'request_type must in [grpc, rest] / current type : {request_type}')
    exit(1)

if edges_to_inference is None:
    edges_to_inference = registered_edges
else:
    edges_to_inference = edges_to_inference.split(',')

for edge in edges_to_inference:
    if edge not in registered_edges:
        print(f'--edge arg must be in {registered_edges}')
        exit(1)
        
print(f'Edges to inference: {edges_to_inference}')


# 추론 요청 할 장비들에서 요청 가능한 모델들
def get_enable_models_to_inference():
    models_to_inference = []

    for edge_name in edges_to_inference:
        edge_info = edges_info.get(edge_name)
        models = edge_info.get('models')
        models_to_inference.extend(models)

    return set(models_to_inference)

models_to_inference = get_enable_models_to_inference()
print(f'Models to inference: {models_to_inference}')


# 추론요청 할 각 모델의 모듈을 저장.
def regist_enable_model_modules(models_to_infer):
    model_modules = {}

    for model in models_to_infer:
        module = None
        if request_type == 'grpc':
            module = importlib.import_module(f"{model}.grpc_bench")
        else:
            module = importlib.import_module(f"{model}.rest_bench")
        model_modules.update({model: module})

    return model_modules

model_modules = regist_enable_model_modules(models_to_inference)
print(f'Model modules: {model_modules}')


# 딕셔너리에 모델별로 엣지장비이름 등록 -> 들어오는 요청에 따라 어느 장비에 보낼 차례인지 확인 할 수 있는 딕셔너리 생성 
def regist_edges_to_model():
    model_edge_info = {}

    for edge in edges_to_inference:
        edge_info = edges_info.get(edge)
        for model in edge_info.get('models'):
            if model not in model_edge_info.keys():
                model_edge_info[model] = []

            model_edge_info[model].append((edge, 1))

    for model in model_edge_info.keys():
        dataset = model_edge_info.get(model)
        model_edge_info[model] = roundrobin.smooth(dataset)

    return model_edge_info

model_edge_info = regist_edges_to_model()
print(f'model-edge dataset: {model_edge_info}')


# 모델로 엣지장비 이름 얻는 함수, 모델을 키값으로 하여 해당 모델이 로드된 엣지장비들을 라운드로빈으로 함수를 호출할 때마다 하나씩 얻어옴
def get_edge_by_model_rr(model):
    if model in model_edge_info.keys():
        return model_edge_info.get(model)()
    else:
        return None


# 추론을 요청하는 함수, 인자로는 추론을 요청할 엣지 장비, 모델. 엣지장비와 모델은 위의 edges_info에 등록되어 있어야함
def model_request(edge, model, idx):
    if edge not in edges_to_inference:
        print(f'edge must be in {edges_to_inference}/ input value: {edge}')
        return

    if model not in models_to_inference:
        print(f'model must be in {models_to_inference}/ input value: {model}')
        return

    edge_info = edges_info.get(edge)
    edge_ip_addr = edge_info.get('ip_addr')

    if request_type == 'grpc':
        edge_ip_addr = f'{edge_ip_addr}:{grpc_port}'
        result = model_modules[model].run_bench(1, edge_ip_addr, 0)
    else:
        edge_ip_addr = f'http://{edge_ip_addr}:{rest_port}/'
        result = model_modules[model].run_bench(1, edge_ip_addr)

    inference_times[idx].extend(result)


# 들어오는 요청들
requests_list = inference_request_workload_manager.load_workload_from_file(workload_file_name)


# 요청을 각 장비에 전달, 여러요청을 동시에 다룰 수 있도록 쓰레드 이용
threads = []
inference_times = [[] for _ in range(len(requests_list))]
idx = -1

start_inference_time = time.time()

for cur_reqs in requests_list:
    idx += 1
    request_sleep_time = 1 / len(cur_reqs)  # 요청들을 1초에 나눠서 보내기 위한 슬립시간
    for req in cur_reqs:
        edge_to_infer = get_edge_by_model_rr(req)
        if edge_to_infer is None:
            print(f'{req} can\'t be inference')
            continue

        th = Thread(target=model_request, args=(edge_to_infer, req, idx))
        th.start()
        threads.append(th)
        time.sleep(request_sleep_time)

for th in threads:
    th.join()

end_inference_time = time.time()
total_inference_time = end_inference_time - start_inference_time

print()
print('total inference time', total_inference_time)
print('----------------------')
print('inference time info (each argument is info about requests per sec)')
print()
for time_info in inference_times:
    print(time_info)
    print()
