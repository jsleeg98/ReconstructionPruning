import torch
from pytorchyolo.models import *
from ptflops import get_model_complexity_info


def run():
    # 원본 모델
    cfg = './YOLOv3/yolov3_0.0001.cfg'  # 압축하지 않은 모델
    model_1 = Darknet(cfg)

    # 실제 압축 모델
    cfg = './KITTI/train_result/original_5/compress/FPGM_0.9/real_pruning_cfg_0.9.cfg'  # 압축한 모델
    model_2 = Darknet(cfg)


    dummy_size = (3, 416, 416)

    macs_1, params_1 = get_model_complexity_info(model_1, dummy_size, as_strings=False, print_per_layer_stat=True,
                                                 verbose=True)
    macs_2, params_2 = get_model_complexity_info(model_2, dummy_size, as_strings=False, print_per_layer_stat=True,
                                                 verbose=True)

    print('computational complexity : ', macs_1)
    print('FLOPs : ', int(macs_1) / 2)
    print('parameters', params_1)

    print('computational complexity : ', macs_2)
    print('FLOPs : ', int(macs_2) / 2)
    print('parameters', params_2)

    print('남은 parameters ratio : {:.2f}%'.format(int(params_2) / int(params_1) * 100))
    print('남은 FLOPs ratio : {:.2f}%'.format((int(macs_2) / 2) / (int(macs_1) / 2) * 100))


if __name__ == '__main__':
    run()