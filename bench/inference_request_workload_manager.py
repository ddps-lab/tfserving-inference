from numpy import random
import pickle as pk

# (modle name, requests per second)
inference_request_info = [
    ('mobilenet_v1', 20),
    ('mobilenet_v2', 2),
    ('inception_v3', 2),
    ('yolo_v5', 1)
]

file_name = 'workload4'
req_time_num = 20

def create_inference_request_workload(req_time):
    requests = [[] for _ in range(req_time)]
    total_req_num = 0
    for (model_name, req_per_sec) in inference_request_info:
        workloads = random.poisson(lam=req_per_sec, size=req_time)
        for idx in range(req_time):
            requests[idx].extend([model_name for _ in range(workloads[idx])])
        total_req_num += sum(workloads)

    for idx in range(req_time):
        random.shuffle(requests[idx])

    workload_info = {}
    workload_info['total_request_num'] = total_req_num
    workload_info['requests'] = requests

    return workload_info


def save_workload_info_to_file(file_name, workloads):
    with open(file_name, 'wb') as f:
        pk.dump(workloads, f)


def load_workload_info_from_file(file_name):
    with open(file_name, 'rb') as f:
        loaded_workloads = pk.load(f)
    return loaded_workloads


# workloads = create_inference_request_workload(req_time_num)
# save_workload_info_to_file(file_name, workloads)
# loaded_workload_info = load_workload_info_from_file(file_name)
# print(loaded_workload_info)
