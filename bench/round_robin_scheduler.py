# from module import put_data_into_sheet
# put_data_into_sheet.put_data(variables.rest_spreadsheet_id, result, variables.num_tasks)

import argparse
import roundrobin
import time
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
    if request_type == 'grpc':
        edge_ip_addr = f'{edge_ip_addr}:{grpc_port}'
        result = model_modules[model].run_bench(1, edge_ip_addr, 0, request_data)
    else:
        edge_ip_addr = f'http://{edge_ip_addr}:{rest_port}/'
        result = model_modules[model].run_bench(1, edge_ip_addr, request_data)
    res_time = time.time()
    

    # 결과 분석을 위한 기록
    inference_times_per_model_in_edge[edge][model].extend(result)
    inference_times_per_sec[idx].extend(result) 
    inference_times_per_edge_and_sec[idx][edge].extend(result)
    inference_times_per_model_in_edge_and_sec[idx][edge][model].extend(result)
    models_per_edge_and_sec[idx][edge].append(model)
    inference_times_all.extend(result)
    req_and_res_time_per_edge[edge].append((req_time, res_time))


# 들어오는 요청들
requests_info = inference_request_workload_manager.load_workload_info_from_file(workload_file_name)
total_req_num = requests_info.get('total_request_num')
requests_list = requests_info.get('requests') 

# 결과 분석을 위한 자료구조들
inference_times_all = [] # 모든 요청에 대한 추론시간을 기록
inference_times_per_sec = [[] for _ in range(len(requests_list))] # 매초 보낸 요청이 처리되는데 걸린 추론시간을 기록
inference_times_per_edge_and_sec = [{edge: [] for edge in edges_to_inference} for _ in range(len(requests_list))] # 
inference_times_per_model_in_edge = {edge: {model: [] for model in models_to_inference} for edge in edges_to_inference} # 각 엣지에서 각 모델이 추론하는데 걸린 시간을 기록 
inference_times_per_model_in_edge_and_sec = [{edge: {model: [] for model in models_to_inference} for edge in edges_to_inference} for _ in range(len(requests_list))] # 매초 보낸 요청이 각 엣지 각 모델에서 추론되는데 걸린 시간을 기록
models_per_edge_and_sec = [{edge: [] for edge in edges_to_inference} for _ in range(len(requests_list))] # 매초 보내는 요청이 어떤 엣지에 대한 요청인지를 기록
req_and_res_time_per_edge = {edge: [] for edge in edges_to_inference} # 요청을 보낸 시각과 응답을 받은 시간을 기록


# 총 추론시간 측정 
start_inference_time = time.time()

# 요청을 각 장비에 전달, 여러요청을 동시에 다룰 수 있도록 쓰레드 이용
threads = []
cur_progress = 0
executor = concurrent.futures.ThreadPoolExecutor(1000)
pre_not_complete_reqs = 0
for cur_reqs_idx, cur_reqs in enumerate(requests_list):
    if len(cur_reqs) <= 0:
        continue

    request_sleep_time = 1 / len(cur_reqs)  # 요청들을 1초에 나눠서 보내기 위한 슬립시간


    # 시나리오에 따라 요청을 전송, 각 요청은 쓰레드로 병렬처리
    cur_reqs_start_time = time.time()
    for req_idx, req in enumerate(cur_reqs):
        edge_to_infer = get_edge_by_model_rr(req)
        if edge_to_infer is None:
            print(f'{req} can\'t be inference')
            continue

        threads.append(executor.submit(model_request, edge_to_infer, req, cur_reqs_idx, cur_progress))
        time.sleep(request_sleep_time)

        cur_progress += 1
        print(f'progress: {cur_progress}/{total_req_num}', end='\r')
    print(f'{cur_reqs_idx+1}초대 모든 요청전송 완료시간 : {time.time() - cur_reqs_start_time}') # 시나리오가 실제로 1초동안 원하는 양만큼 전송하는지 확인하는 코드


    # 매초마다 처리되지 않은 요청량과 이전의 보냈던 요청처리량, 현재 시점에 보낸 요청처리량을 출력
    completed_req_num = 0
    real_req_num = 0
    real_req_num = 0
    for i in range(cur_reqs_idx+1):
        real_req_num += len(requests_list[i])
        completed_req_num += len(inference_times_per_sec[i])
        if i == cur_reqs_idx-1 and cur_reqs_idx != 0:
            pre_completed_req_num = completed_req_num
    print(f'{cur_reqs_idx+1}초대 완료되지 않은 요청량: {real_req_num - completed_req_num}/{real_req_num}', f'{len(cur_reqs)+pre_completed_req_num-completed_req_num}/{len(cur_reqs)}')
    print(f'{cur_reqs_idx}초대 완료된 요청량(완료되지않은 요청량): {pre_completed_req_num}/{real_req_num-len(cur_reqs)}({real_req_num-len(cur_reqs)-pre_completed_req_num})')
    print(f'{cur_reqs_idx+1}초대 처리량: {pre_not_complete_reqs+len(cur_reqs) - (real_req_num - completed_req_num)}')
    pre_not_complete_reqs = real_req_num - completed_req_num

print(f'waiting to complete...')

for th in concurrent.futures.as_completed(threads):
    continue

print('complete!')

# 총 추론시간 측정 
end_inference_time = time.time()
total_inference_time = end_inference_time - start_inference_time


# 테스트 결과를 출력

print()
print('total request num: ', total_req_num)
print('total inference time', total_inference_time)
print()
print('----------------------')
print('inference time info (each argument is info about requests per sec)')
print()


# 시나리오 전체에 대한 추론결과를 출력 (총처리량, 평균 최소 최대 처리시간 등)
def print_overall_inference_result():
    inference_times_all.sort()
    inference_times_len = len(inference_times_all)
    print(f'reqeust num per sec: ', inference_times_len)
    print(f'avg:', sum(inference_times_all)/inference_times_len)
    print(f'min:', inference_times_all[0])
    print(f'max:', inference_times_all[-1])

    print(f'25%:', inference_times_all[int(inference_times_len/4)])
    print(f'50%:', inference_times_all[int(inference_times_len/2)])
    print(f'75%:', inference_times_all[int((inference_times_len*3)/4)])

    print()


# 시나리오에서 매초 처리한 처리량, 평균 최소 최대 처리시간 등 출력
def print_inference_result_per_sec():
    for i, time_info in enumerate(inference_times_per_sec, start=1):
        time_info.sort()
        time_info_len = len(time_info)

        print(f'[{i}] reqeust num per sec: ', time_info_len)
        if time_info_len == 0:
            print()
            continue
        print(f'[{i}] avg:', sum(time_info)/time_info_len)
        print(f'[{i}] min:', time_info[0])
        print(f'[{i}] max:', time_info[-1])

        print(f'[{i}] 25%:', time_info[int(time_info_len/4)])
        print(f'[{i}] 50%:', time_info[int(time_info_len/2)])
        print(f'[{i}] 75%:', time_info[int((time_info_len*3)/4)])

        print()


# 리스트안의 각 요소의 개수를 세는 함수
def count_each_element_in_list(list_to_count: list):
    element_names = list(set(list_to_count))
    element_names.sort()
    result = []
    for name in element_names:
        result.append(f'{name}: {list_to_count.count(name)}')

    return result


# 엣지마다 요청이 들어온 모델을 기준으로 처리량, 평균 최소 최대 처리시간 등 출력
def print_inference_result_per_model_in_edge():
    for edge in inference_times_per_model_in_edge:
        models = inference_times_per_model_in_edge[edge]
        print(edge)
        for model in models:
            infer_time_per_model = models[model]
            infer_time_per_model.sort()
            infer_time_per_model_len = len(infer_time_per_model)
            total_infer_time_per_model = sum(infer_time_per_model)

            if infer_time_per_model_len == 0:
                print(f'{model} is not infered')
                continue
            print(f'[{model}] request num:', infer_time_per_model_len)
            print(f'[{model}] total:', total_infer_time_per_model)
            print(f'[{model}] avg:', total_infer_time_per_model/infer_time_per_model_len)
            print(f'[{model}] min:', infer_time_per_model[0])
            print(f'[{model}] max:', infer_time_per_model[-1])

            print(f'[{model}] 25%:', infer_time_per_model[int(infer_time_per_model_len/4)])
            print(f'[{model}] 50%:', infer_time_per_model[int(infer_time_per_model_len/2)])
            print(f'[{model}] 75%:', infer_time_per_model[int((infer_time_per_model_len*3)/4)])
        print()


# 엣지에서 매초 처리한 처리량, 평균 최소 최대 처리시간 등 출력
def print_inference_result_per_edge_and_sec():
    for i, time_info in enumerate(inference_times_per_edge_and_sec, start=1):
        for edge in time_info.keys():
            times = time_info.get(edge)

            times.sort()
            times_len = len(times)

            models = models_per_edge_and_sec[i-1].get(edge)
            requests = count_each_element_in_list(models)

            print(f'[{i}초대 {edge} 장비] reqeust num per sec: ', times_len)
            if times_len == 0:
                print()
                continue
            print(f'[{i}초대 {edge} 장비] requests: ', requests)
            print(f'[{i}초대 {edge} 장비] avg:', sum(times)/times_len)
            print(f'[{i}초대 {edge} 장비] min:', times[0])
            print(f'[{i}초대 {edge} 장비] max:', times[-1])

            print(f'[{i}초대 {edge} 장비] 25%:', times[int(times_len/4)])
            print(f'[{i}초대 {edge} 장비] 50%:', times[int(times_len/2)])
            print(f'[{i}초대 {edge} 장비] 75%:', times[int((times_len*3)/4)])

            print()


# 매초 엣지마다 요청이 들어온 모델을 기준으로 처리량, 평균 최소 최대 처리시간 등 출력
def print_inference_result_per_model_in_edge_and_sec():
    for i, time_info in enumerate(inference_times_per_model_in_edge_and_sec, start=1):
        for edge in time_info.keys():
            times_per_model_info = time_info.get(edge)
            times = []

            for model in times_per_model_info.keys():
                times_per_model = times_per_model_info.get(model)
                times_per_model.sort()
                times_per_model_len = len(times_per_model)

                print(f'[{i}초대 {edge} 장비 {model} 모델] request num per sec:', times_per_model_len)
                if times_per_model_len == 0:
                    print()
                    continue
                print(f'[{i}초대 {edge} 장비 {model} 모델] avg:', sum(times_per_model)/times_per_model_len)
                print(f'[{i}초대 {edge} 장비 {model} 모델] min:', times_per_model[0])
                print(f'[{i}초대 {edge} 장비 {model} 모델] max:', times_per_model[-1])
                times.extend(times_per_model)
                print()

            times.sort()
            times_len = len(times)

            models = models_per_edge_and_sec[i-1].get(edge)
            requests = count_each_element_in_list(models)

            print(f'[{i}초대 {edge} 장비] reqeust num per sec: ', times_len)
            if times_len == 0:
                print()
                continue
            print(f'[{i}초대 {edge} 장비] requests: ', requests)
            print(f'[{i}초대 {edge} 장비] avg:', sum(times)/times_len)
            print(f'[{i}초대 {edge} 장비] min:', times[0])
            print(f'[{i}초대 {edge} 장비] max:', times[-1])

            print(f'[{i}초대 {edge} 장비] 25%:', times[int(times_len/4)])
            print(f'[{i}초대 {edge} 장비] 50%:', times[int(times_len/2)])
            print(f'[{i}초대 {edge} 장비] 75%:', times[int((times_len*3)/4)])

            print()

print_overall_inference_result()
print_inference_result_per_sec()
print_inference_result_per_model_in_edge()
print_inference_result_per_edge_and_sec()
print_inference_result_per_model_in_edge_and_sec()


print('----------------------')
print('idle time by edge (measurement including request processing time, network delay)')
print()


# idle time 출력, 실제로 장비의 유휴시간이 아닌 네트워크 지연을 포함
def print_idle_time_by_edge():
    for edge in req_and_res_time_per_edge:
        req_time_info = req_and_res_time_per_edge.get(edge)
        req_time_info.sort()

        cur_req_time = 0
        cur_res_time = 0

        idle_time = []
        idle_time_info = []

        for (req_time, res_time) in req_time_info:
            if cur_req_time == 0:
                idle_time.append(req_time - start_inference_time)
                idle_time_info.append((start_inference_time - start_inference_time, req_time - start_inference_time))
                cur_req_time = req_time
                cur_res_time = res_time
            elif req_time >= cur_req_time and req_time <= cur_res_time:
                if res_time > cur_res_time:
                    cur_res_time = res_time
            else:
                idle_time.append(req_time - cur_res_time)
                idle_time_info.append((cur_res_time - start_inference_time, req_time - start_inference_time))
                cur_req_time = req_time
                cur_res_time = res_time
        if cur_res_time != 0:
            idle_time.append(end_inference_time - cur_res_time)
            idle_time_info.append((cur_res_time - start_inference_time, end_inference_time - start_inference_time))

        idle_time.sort()
        idle_time_len = len(idle_time)
        total_idle_time = sum(idle_time)

        requests_per_edge = []
        for ed in models_per_edge_and_sec:
            requests_per_edge.extend(ed.get(edge))
        requests = count_each_element_in_list(requests_per_edge)
        print(f'[{edge}] requests num:', len(req_time_info))
        if idle_time_len == 0:
            print()
            continue
        print(f'[{edge}] requests:', requests)
        print(f'[{edge}] total idle time:', total_idle_time)

        idle_time_percent = []
        for (s, e) in idle_time_info:
            idle_time_percent.append(f'{((e - s) / total_inference_time) * 100}%')

        print('total inference time:', total_inference_time)
        print('idle time info:', idle_time_info)
        print('idle time:', idle_time)
        print('idle time percent:', idle_time_percent)

        print()

print_idle_time_by_edge()