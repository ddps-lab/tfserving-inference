# from module import put_data_into_sheet
# put_data_into_sheet.put_data(variables.rest_spreadsheet_id, result, variables.num_tasks)

import argparse
import roundrobin
import time
from threading import Thread
import concurrent.futures
import importlib
import inference_request_workload_manager
import preprocessing_data_manager


parser = argparse.ArgumentParser()
parser.add_argument('--edge', default=None, type=str)
parser.add_argument('--workload', type=str, required=True)

scheduler_args = parser.parse_args()
edges_to_inference = scheduler_args.edge

# grpc or rest
request_type = 'rest'
# workload file name
workload_file_name = scheduler_args.workload
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
                               'models': ['mobilenet_v1', 'mobilenet_v2', 'inception_v3']
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


# 각 모델의 preprocessing 데이터 저장
preprocessed_datas = preprocessing_data_manager.regist_preprocessed_datas(request_type)
print(f'Preprocessed datas: {preprocessed_datas}')


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
def model_request(edge, model, idx, thread_id):
    if edge not in edges_to_inference:
        print(f'edge must be in {edges_to_inference}/ input value: {edge}')
        return

    if model not in models_to_inference:
        print(f'model must be in {models_to_inference}/ input value: {model}')
        return

    edge_info = edges_info.get(edge)
    edge_ip_addr = edge_info.get('ip_addr')
    request_data = preprocessed_datas.get(model)

    req_time = time.time()
    print(f'[{thread_id}] req start: {req_time}')
    if request_type == 'grpc':
        edge_ip_addr = f'{edge_ip_addr}:{grpc_port}'
        result = model_modules[model].run_bench(1, edge_ip_addr, 0, request_data)
    else:
        edge_ip_addr = f'http://{edge_ip_addr}:{rest_port}/'
        result = model_modules[model].run_bench(1, edge_ip_addr, request_data)

    res_time = time.time()
    print(f'[{thread_id}] req end: {res_time}')
    #print(f'[{thread_id}] total: {res_time - req_time}')
    inference_times[idx].extend(result) # 추론시간 기록
    # inference_times.extend(result) # 추론시간 기록
    edge_req_time_info[edge].append((req_time, res_time))


# 들어오는 요청들
requests_info = inference_request_workload_manager.load_workload_info_from_file(workload_file_name)
total_req_num = requests_info.get('total_request_num')
requests_list = requests_info.get('requests')


# 요청을 각 장비에 전달, 여러요청을 동시에 다룰 수 있도록 쓰레드 이용
threads = []
# inference_times = []
inference_times = [[] for _ in range(len(requests_list))]

edge_req_time_info = {}
for edge in edges_to_inference:
    edge_req_time_info[edge] = []

start_inference_time = time.time()

# cur_progress = 0
# for cur_reqs_idx, cur_reqs in enumerate(requests_list):
#     request_sleep_time = 1 / len(cur_reqs)  # 요청들을 1초에 나눠서 보내기 위한 슬립시간

#     # cur_reqs_start_time = time.time()
#     for req_idx, req in enumerate(cur_reqs):
#         edge_to_infer = get_edge_by_model_rr(req)
#         if edge_to_infer is None:
#             print(f'{req} can\'t be inference')
#             continue
        
#         # thread_start_time = time.time()
        
#         th = Thread(target=model_request, args=(edge_to_infer, req, cur_reqs_idx, cur_progress))
#         th.start()
#         threads.append(th)

#         # thread_complete_time = time.time() - thread_start_time
#         # if thread_complete_time < request_sleep_time:
#         #     time.sleep(request_sleep_time - thread_complete_time)
#         cur_progress += 1
#         print(f'progress: {cur_progress}/{total_req_num}', end='\r')

#     # cur_reqs_end_time = time.time()
#     # cur_reqs_total_time = cur_reqs_end_time - cur_reqs_start_time
#     # if cur_reqs_total_time < 1:
#     #     time.sleep(1 - cur_reqs_total_time)

# print(f'waiting to complete... (it takes about {cur_reqs_idx+1} seconds)')

# for th in threads:
#     th.join()

cur_progress = 0
executor = concurrent.futures.ThreadPoolExecutor(1000)
for cur_reqs_idx, cur_reqs in enumerate(requests_list):
    request_sleep_time = 1 / len(cur_reqs)  # 요청들을 1초에 나눠서 보내기 위한 슬립시간

    cur_reqs_start_time = time.time()
    for req_idx, req in enumerate(cur_reqs):
        edge_to_infer = get_edge_by_model_rr(req)
        if edge_to_infer is None:
            print(f'{req} can\'t be inference')
            continue
        
        # thread_start_time = time.time()
        
        executor.submit(model_request, edge_to_infer, req, cur_reqs_idx, cur_progress)
    
        cur_progress += 1
        print(f'progress: {cur_progress}/{total_req_num}', end='\r')

    cur_reqs_end_time = time.time()
    cur_reqs_total_time = cur_reqs_end_time - cur_reqs_start_time
    if cur_reqs_total_time < 1:
        time.sleep(1 - cur_reqs_total_time)

print(f'waiting to complete... (it takes about {cur_reqs_idx+1} seconds)')



# cur_progress = 0
# s = sched.scheduler()
# for cur_reqs_idx, cur_reqs in enumerate(requests_list):
#     request_sleep_time = 1 / len(cur_reqs)  # 요청들을 1초에 나눠서 보내기 위한 슬립시간
#     for req_idx, req in enumerate(cur_reqs):
#         edge_to_infer = get_edge_by_model_rr(req)
#         if edge_to_infer is None:
#             print(f'{req} can\'t be inference')
#             continue
        
#         s.enter((request_sleep_time * req_idx) + cur_reqs_idx, 1, model_request, (edge_to_infer, req, cur_reqs_idx, cur_progress,))
#         cur_progress += 1
#         print(f'progress: {cur_progress}/{total_req_num}', end='\r')

# print(f'waiting to complete... (it takes about {cur_reqs_idx-1} seconds)')

# s.run()

print('complete!')

end_inference_time = time.time()
total_inference_time = end_inference_time - start_inference_time

print()
print('total request num: ', total_req_num)
print('total inference time', total_inference_time)
print()
print('----------------------')
print('inference time info (each argument is info about requests per sec)')
print()

for i, time_info in enumerate(inference_times, start=1):
    time_info.sort()
    time_info_len = len(time_info)

    print(f'[{i}] reqeust num per sec: ', time_info_len)
    print(f'[{i}] avg:', sum(time_info)/time_info_len)
    print(f'[{i}] min:', time_info[0])
    print(f'[{i}] max:', time_info[-1])

    print(f'[{i}] 25%:', time_info[int(time_info_len/4)])
    print(f'[{i}] 50%:', time_info[int(time_info_len/2)])
    print(f'[{i}] 75%:', time_info[int((time_info_len*3)/4)])
    
    print()

# inference_times.sort()
# inference_times_len = len(inference_times)
# print(f'reqeust num per sec: ', inference_times_len)
# print(f'avg:', sum(inference_times)/inference_times_len)
# print(f'min:', inference_times[0])
# print(f'max:', inference_times[-1])

# print(f'25%:', inference_times[int(inference_times_len/4)])
# print(f'50%:', inference_times[int(inference_times_len/2)])
# print(f'75%:', inference_times[int((inference_times_len*3)/4)])

print('----------------------')
print('idle time by edge')
print()


for edge in edge_req_time_info:
    req_time_info = edge_req_time_info.get(edge)
    req_time_info.sort()

    cur_req_time = 0
    cur_res_time = 0

    idle_time = []

    for (req_time, res_time) in req_time_info:
        if cur_req_time == 0:
            idle_time.append(req_time - start_inference_time)
            cur_req_time = req_time
            cur_res_time = res_time
        elif req_time >= cur_req_time and req_time <= cur_res_time:
            if res_time > cur_res_time:
                cur_res_time = res_time
        else:
            idle_time.append(req_time - cur_res_time)
            cur_req_time = req_time
            cur_res_time = res_time
    if cur_res_time != 0:
        idle_time.append(end_inference_time - cur_res_time)

    idle_time.sort()
    idle_time_len = len(idle_time)
    total_idle_time = sum(idle_time)

    if idle_time_len == 0:
        print(f'{edge}\'s idle time is 0')
        print()
        continue

    print(f'[{edge}] total:', total_idle_time)
    print(f'[{edge}] avg:', total_idle_time/idle_time_len)
    print(f'[{edge}] min:', idle_time[0])
    print(f'[{edge}] max:', idle_time[-1])

    print(f'[{edge}] 25%:', idle_time[int(idle_time_len/4)])
    print(f'[{edge}] 50%:', idle_time[int(idle_time_len/2)])
    print(f'[{edge}] 75%:', idle_time[int((idle_time_len*3)/4)])   

    print()
