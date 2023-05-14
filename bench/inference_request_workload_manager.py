from numpy import random
import pickle as pk

# (modle name, requests per second)
inference_request_info = [
    ('mobilenet_v1', 10),
    ('mobilenet_v2', 2),
    ('inception_v3', 2),
    ('yolo_v5', 1)
]

file_name = 'inference_request_workload.pickle'


def create_inference_request_workload(req_time):
    requests = [[] for _ in range(req_time)]
    for (model_name, req_per_sec) in inference_request_info:
        workloads = random.poisson(lam=req_per_sec, size=req_time)
        for idx in range(req_time):
            requests[idx].extend([model_name for _ in range(workloads[idx])])

    for idx in range(req_time):
        random.shuffle(requests[idx])

    return requests


def save_workload_to_file(file_name, workloads):
    with open(file_name, 'wb') as f:
        pk.dump(workloads, f)


def load_workload_from_file(file_name):
    with open(file_name, 'rb') as f:
        loaded_workloads = pk.load(f)
    return loaded_workloads


# workloads = create_inference_request_workload(100)

# save_workload_to_file(file_name, workloads)
# loaded_workloads = load_workload(file_name)


# for i in loaded_workloads:
#     print(i)
#     print()