import time
import numpy as np
from PIL import Image
#REST Import
import requests
import json
import concurrent.futures

serving_url = "http://localhost:8501/v1/models/"
model_name = "mobilenet_v1"
image_file_path = "../../../dataset/imagenet/imagenet_1000_raw/n01843383_1.JPEG"

def send_request(img_array,data):
    start_time = time.time()
    headers = {"content-type": "application/json"}
    url = serving_url + model_name + ":predict"
    response = requests.post(url, data=data, headers=headers)
    end_time = time.time()
    result = json.loads(response.text)
    return result, end_time - start_time

img = Image.open(image_file_path)
img = img.resize((224, 224))
img_array = np.array(img)
img_array = img_array.astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)
data = json.dumps({"instances": img_array.tolist()})

n_requests = 100  # 요청 횟수 변수 선언

start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as executor:
    futures = [executor.submit(send_request, img_array, data) for _ in range(n_requests)]
    results = []
    for future in concurrent.futures.as_completed(futures):
        result, thread_time = future.result()
        results.append(result)
        print("Thread execution time: ", thread_time)

end_time = time.time()
response_time = (end_time - start_time)

print("Total response time: ", response_time)
print(results)

