import tensorflow as tf
import json
import importlib


data_source_info = {'mobilenet_v1': '../dataset/imagenet/imagenet_1000_raw/n01843383_1.JPEG',
                    'mobilenet_v2': '../dataset/imagenet/imagenet_1000_raw/n01843383_1.JPEG',
                    'inception_v3': '../dataset/imagenet/imagenet_1000_raw/n01843383_1.JPEG',
                    'yolo_v5': '../dataset/coco_2017/coco/images/val2017/000000089761.jpg',
                }

def regist_preprocessed_datas(request_type):
    preprocessed_datas = {}

    for model in data_source_info.keys():
        if request_type == 'rest':
            preprocessing_module = importlib.import_module(f"{model}.preprocessing")
            data_source = data_source_info.get(model)
            data = json.dumps({"instances": preprocessing_module.run_preprocessing(data_source).tolist()})
            preprocessed_datas.update({model: data})
        elif request_type == 'grpc':
            preprocessing_module = importlib.import_module(f"{model}.preprocessing")
            data_source = data_source_info.get(model)
            data = tf.make_tensor_proto(preprocessing_module.run_preprocessing(data_source))
            preprocessed_datas.update({model: data})

    return preprocessed_datas