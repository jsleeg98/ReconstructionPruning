'''
FPGM에서conv에서 0이 되는 인덱스와 동일한 인덱스를 bn_bias에서도 0이 되도록 수정
'''

# ! /usr/bin/env python3


from __future__ import division

import os
import argparse
import tqdm
import sys
sys.path.append('/home/openlab/DH_Lee/ReconstructionPruning/')

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from pytorchyolo.models import *
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary

import pandas as pd
import matplotlib.pyplot as plt

##FPGM-------------------------------#
import random
from scipy.spatial import distance
import numpy as np
from collections import OrderedDict

# -----------------------------------#
# cfg 생성 시 활용
def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')  # 줄바꿈 단위로 나눈다.
    lines = [x for x in lines if x and not x.startswith('#')]  # 라인에 글이 있고 #으로 시작하지 않으면 line으로 list화한다.
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces #각 줄에서 앞뒤 공백은 모두 제거한다.
    module_defs = []
    for line in lines:

        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
            # 수정
            if module_defs[-1]['type'] == 'predict':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


class Mask:
    def __init__(self, model, device):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}
        self.device = device

        # 추가
        self.zero_index = {}

    # 쓰이지 않음
    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")

        return weight_np

    # compression_rate : norm 기준으로 잘라줄 filter 비율
    # length :
    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            # print("filter codebook done")
        else:
            pass
        return codebook

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter index done")
        else:
            pass
        return filter_small_index, filter_large_index

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)

            # norm 값을 이용하여 필터를 자르는 부분으로 실질적으로 자르는데 시용하지 않는다.
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # Geometric Median 값을 이용하여 실질적으로 자르는 부분
            # distance using numpy function
            # indices = torch.LongTensor(filter_large_index).cuda()
            indices = torch.LongTensor(filter_large_index).to(self.device)  # 수정
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()

            # GM과의 거리 계산하는 부분
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]
            # similar_index_for_filter는 0이 되는 인덱스 저장

            # print('filter_large_index', filter_large_index)
            # print('filter_small_index', filter_small_index)
            # print('similar_sum', similar_sum)
            # print('similar_large_index', similar_large_index)
            # print('similar_small_index', similar_small_index)
            # print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            # print("similar index done")
        else:
            pass

        # 수정
        # similar_index_for_filter를 이용하여 bn_weight와 bn_bias도 0으로 변경해야하기 때문에 같이 리턴으로 수정
        return codebook, similar_index_for_filter

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
            self.distance_rate[index] = 1
        # layer_begin, layer_end, layer_inter 임의 설정
        # 수정
        # layer_begin = 0
        # layer_end = 228
        # layer_inter = 3

        cnt = 0
        for name, param in self.model.named_parameters():
            if len(param.size()) == 4 and not ('pred' in name):
                # if len(param.size()) == 4 :
                self.compress_rate[cnt] = rate_norm_per_layer
                self.distance_rate[cnt] = rate_dist_per_layer
                self.mask_index.append(cnt)
            cnt = cnt + 1
        # for key in range(layer_begin, layer_end + 1, layer_inter):
        #     self.compress_rate[key] = rate_norm_per_layer
        #     self.distance_rate[key] = rate_dist_per_layer

        # different setting for  different architecture
        # if args.arch == 'resnet20':
        #     last_index = 57
        # elif args.arch == 'resnet32':
        #     last_index = 93
        # elif args.arch == 'resnet56':
        #     last_index = 165
        # elif args.arch == 'resnet110':
        #     last_index = 327
        # to jump the last fc layer

        # 수정
        # self.mask_index = [x for x in range(0, layer_end, 3)]

        #    self.mask_index =  [x for x in range (0,330,3)]

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # norm의 크기로 필터를 자르는 것으로 우리는 사용하지 않는다.
                # mask for norm criterion
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])

                self.mat[index] = self.convert2tensor(self.mat[index])
                # gpu 사용
                # if args.use_cuda:
                # self.mat[index] = self.mat[index].cuda()
                self.mat[index] = self.mat[index].to(self.device)  # 수정
                # get result about filter index
                self.filter_small_index[index], self.filter_large_index[index] = \
                    self.get_filter_index(item.data, self.compress_rate[index], self.model_length[index])

                # Geometric Median 기법을 이용하여 필터 자름 -> 사용
                # mask for distance criterion
                self.similar_matrix[index], zero_index = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                                 self.distance_rate[index],
                                                                                 self.model_length[index],
                                                                                 dist_type=dist_type)

                # 수정
                # print(zero_index)
                for temp_index, temp_item in enumerate(self.model.parameters()):
                    # bn_bias에서도 해당 인덱스를 0으로 만들기
                    if index + 1 == temp_index:
                        temp_tensor = torch.ones_like(temp_item.data)
                        for i in zero_index:
                            temp_tensor[i] = 0
                        temp_item.data = temp_item.data * temp_tensor
                    if index + 2 == temp_index:
                        temp_tensor = torch.ones_like(temp_item.data)
                        for i in zero_index:
                            temp_tensor[i] = 0
                        temp_item.data = temp_item.data * temp_tensor
                        break

                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])

                # gpu 사용
                # if args.use_cuda:
                # self.similar_matrix[index] = self.similar_matrix[index].cuda()
                self.similar_matrix[index] = self.similar_matrix[index].to(self.device)  # 수정
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])

        print("mask Done")

    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])

        print("mask similar Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                b = a * self.mat[index]
                b = b * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        # print("grad zero Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                # print(
                #     "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="./KITTI/data/KITTI_5.data",
                        help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=16, help="Number of cpu threads to use during batch generation")
    # parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1,
                        help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.2, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.45,
                        help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=777, help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument("-c", "--compression", type=float, default=1.0)
    parser.add_argument("-g", "--gpu", type=str, default='cuda:0')
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)  # Tensorboard logger

    # Get data configuration
    data_config = parse_data_config(args.data)
    # import pdb; pdb.set_trace()
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # gpu 선택 설정
    cuda_num = args.gpu
    print('gpu : {}'.format(cuda_num))
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

    # FPGM 압축률 설정
    FPGM_compress_rate = args.compression
    print('압축률 : {}'.format(str(FPGM_compress_rate)))

    # 결과 폴더 생성
    os.makedirs("./KITTI/train_result/original_5/compress/FPGM_{}".format(str(FPGM_compress_rate)), exist_ok=True)
    os.makedirs("./KITTI/train_result/original_5/compress/FPGM_{}/graph".format(str(FPGM_compress_rate)),
                exist_ok=True)
    os.makedirs("./KITTI/train_result/original_5/compress/FPGM_{}/weight".format(str(FPGM_compress_rate)),
                exist_ok=True)

    result_path = './KITTI/train_result/original_5/compress/FPGM_{}'.format(str(FPGM_compress_rate))

    # ############
    # Create model
    # ############

    # model = load_model(args.model, args.pretrained_weights)

    # 원래 retrain
    cfg = './YOLOv3/yolov3_0.0001.cfg'
    model = Darknet(cfg).to(device)

    # weight_decay 0.0005
    # model.load_state_dict(torch.load('./FPGM/weight/yolov3_original_0.54.pth', map_location = device))

    # weight_decay 0.0001
    model.load_state_dict(
        torch.load('./KITTI/train_result/original_5/weight/yolov3_train_best_fold5.pth', map_location=device))

    # 압축 retrain
    # cfg = './FPGM/test_same.cfg' # 압축한 모델
    # model = Darknet(cfg).to(device)
    # model.apply(weights_init_normal)
    # model.load_state_dict(torch.load('test_same_final.pth'))

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

    # first test
    print("\n---- Evaluating Model ----")
    print("가져온 모델 성능 평가")
    # Evaluate the model on the validation set
    metrics_output = _evaluate(
        model,
        validation_dataloader,
        class_names,
        img_size=model.hyperparams['height'],
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=args.verbose,
        device=device
    )

    precision, recall, AP, f1, ap_class = metrics_output

    # 수정
    # 결과 출력
    print("validation/precision : {}".format(precision.mean()))
    print("validation/recall : {}".format(recall.mean()))
    print("validation/mAP : {}".format(AP.mean()))
    print("validation/f1 : {}".format(f1.mean()))

    # Mask 클래스 객체 초기화, 딕셔너리, 리스트 선언과 self.model = model 설정
    m = Mask(model, device)
    # model_size 크기 저장
    m.init_length()

    m.model = model
    m.init_mask(rate_norm_per_layer=1, rate_dist_per_layer=FPGM_compress_rate, dist_type='l2')
    # m.do_mask() # norm 크기에 따라 자르기 때문에 사용하지 않음
    m.do_similar_mask()
    model = m.model
    model.to(device)

    # torch.save(model.state_dict(), './FPGM/weight/test_0.1.pth')

    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param)
    #     import pdb; pdb.set_trace()

    mm = 0
    for name, param in model.named_parameters():

        li = list(param.size())
        # if len(li) == 4:
        print(name, end=" : ")
        print(param.size())
        cnt = 0
        for i in range(li[0]):
            if torch.norm(param[i]) == 0:
                # print(i, end = ' : ')
                # print(torch.norm(param[i]))
                cnt += 1
        print(cnt)
        mm += 1

    print(mm)

    # first test
    print("\n---- Evaluating Model ----")
    print("FPGM 적용 후 retrain전 테스트")
    # Evaluate the model on the validation set
    metrics_output = _evaluate(
        model,
        validation_dataloader,
        class_names,
        img_size=model.hyperparams['height'],
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=args.verbose,
        device=device
    )

    precision, recall, AP, f1, ap_class = metrics_output

    # 수정
    # 결과 출력
    print("validation/precision : {}".format(precision.mean()))
    print("validation/recall : {}".format(recall.mean()))
    print("validation/mAP : {}".format(AP.mean()))
    print("validation/f1 : {}".format(f1.mean()))

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 150, eta_min = 0.000001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,120], gamma=0.5)

    # epoch 저장 리스트
    li_epoch = []
    # loss 저장 리스트
    li_loss = []
    # MAP 저장 리스트
    li_MAP = []
    # Best MAP 저장
    best_MAP = 0

    for epoch in range(args.epochs):
        # epoch list 추가
        li_epoch.append(epoch + 1)

        print("\n---- Training Model ----")
        # lr 출력
        print('learning rate : {}'.format(optimizer.param_groups[0]['lr']))

        model.train()  # Set model to training mode

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            # print(to_cpu(loss).item())
            ###############
            # Run optimizer
            ###############

            # if batches_done % model.hyperparams['subdivisions'] == 0:
            #     # Adapt learning rate
            #     # Get learning rate defined in cfg
            #     lr = model.hyperparams['learning_rate']
            #     if batches_done < model.hyperparams['burn_in']:
            #         # Burn in
            #         lr *= (batches_done / model.hyperparams['burn_in'])
            #     else:
            #         # Set and parse the learning rate to the steps defined in the cfg
            #         for threshold, value in model.hyperparams['lr_steps']:
            #             if batches_done > threshold:
            #                 lr *= value
            #     # Log the learning rate
            #     logger.scalar_summary("train/learning_rate", lr, batches_done)

            # # Set learning rate
            # for g in optimizer.param_groups:
            #     g['lr'] = lr

            # Run optimizer
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # 수정
        # lr scheduler
        scheduler.step()

        # loss list에 추가
        li_loss.append(to_cpu(loss).item())

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        # if epoch % args.checkpoint_interval == 0:
        #     checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
        #     print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
        #     torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########

        m.model = model
        # m.if_zero()
        m.init_mask(rate_norm_per_layer=1, rate_dist_per_layer=FPGM_compress_rate, dist_type='l2')
        # m.do_mask() # norm 크기에 따라 자르기 때문에 사용하지 않음
        m.do_similar_mask()
        # m.if_zero()
        model = m.model
        model.to(device)

        # print zero filter
        for name, param in model.named_parameters():
            li = list(param.size())
            # if len(li) == 4:
            print(name, end=" : ")
            print(param.size())
            cnt = 0
            for i in range(li[0]):
                if torch.norm(param[i]) == 0:
                    # print(i, end = ' : ')
                    # print(torch.norm(param[i]))
                    cnt += 1
            print(cnt)

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose,
                device=device
            )
            # 수정
            print('Best mAP : {}'.format(best_MAP))

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # 수정
            # 결과 출력
            print("validation/precision : {}".format(precision.mean()))
            print("validation/recall : {}".format(recall.mean()))
            print("validation/mAP : {}".format(AP.mean()))
            print("validation/f1 : {}".format(f1.mean()))

            # 수정
            li_MAP.append(AP.mean())
            # 최고 성능 모델 저장
            if AP.mean() > best_MAP:
                checkpoint_path = result_path + f'/weight/yolov3_FPGM_best_{str(FPGM_compress_rate)}.pth'
                print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
                torch.save(model.state_dict(), checkpoint_path)
                best_MAP = AP.mean()

        # 현재 학습된 내용 DF 변환
        df = pd.DataFrame([x for x in zip(li_epoch, li_loss, li_MAP)])
        df.columns = ['epoch', 'loss', 'MAP']
        df.to_csv(result_path + f'/graph/yolov3_FPGM_{str(FPGM_compress_rate)}.csv', index=False)

        # 현재 학습된 내용 graph 변환
        plt.figure(1)
        plt.plot(li_epoch, li_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss_graph')
        plt.savefig(result_path + f'/graph/yolov3_FPGM_train_loss_graph_{str(FPGM_compress_rate)}.png')

        # 현재 test한 MAP graph 변환
        plt.figure(2)
        plt.plot(li_epoch, li_MAP)
        plt.xlabel('Epoch')
        plt.ylabel('MAP')
        plt.title('MAP_graph')
        plt.savefig(result_path + f'/graph/yolov3_FPGM_test_MAP_graph_{str(FPGM_compress_rate)}.png')


    # real pruning
    # 실행 설정 값
    # -----------------------------------------------------------------------------#

    # FPGM 압축된 pth
    pretrained = result_path + f'/weight/yolov3_FPGM_best_{str(FPGM_compress_rate)}.pth'
    # 생성된 실제 압축된 pth
    # real_compressed_pth = './FPGM/weight/weight_decay/0.001/yolov3_real_compressed_0.9.pth'
    real_compressed_pth = result_path + f'/weight/real_pruning_weight_{str(FPGM_compress_rate)}.pth'

    # 원본 cfg
    cfg_origin = cfg
    # 생성될 cfg
    cfg_compressed = result_path + f'/real_pruning_cfg_{str(FPGM_compress_rate)}.cfg'

    # ------------------------------------------------------------------------------#

    # non_zero_state_dict 얻기
    # -------------------------------------------------------------------------------#
    device = 'cpu'

    checkpoint = torch.load(pretrained, map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])

    state_dict = OrderedDict()
    non_zero_index_dict = OrderedDict()

    cnt = 0
    non_zero_index = torch.tensor([0, 1, 2])
    for name, param in checkpoint.items():
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
            continue

        # print(name + ' -> ' + str(cnt))
        # print(param.size())

        li_name = name.split('.')

        if not ('pred' in name):
            if 'conv' in li_name[-2]:
                # conv 인 경우
                li = list(param.size())
                a = []
                for i in range(li[0]):
                    a.append(torch.index_select(param[i], 0, non_zero_index))
                tensor = torch.stack(a, 0)

                viewed_param = tensor.view(tensor.size()[0], -1)
                norm = torch.norm(viewed_param, 2, 1)  # norm 변환
                non_zero_index = torch.nonzero(norm)
                non_zero_index = non_zero_index.squeeze()
                layer_state_dict = torch.index_select(tensor, 0, non_zero_index)
                state_dict['conv_{}'.format(cnt)] = layer_state_dict
                non_zero_index_dict['non_zero_index_{}'.format(cnt)] = non_zero_index
            elif 'batch_norm' in li_name[-2] and li_name[-1] == 'bias':
                cnt += 1


        elif 'pred' in name:
            if li_name[-1] == 'weight' and not ('batch_norm' in li_name[-2]):
                # pred의 conv인 경우
                li = list(param.size())
                a = []
                for i in range(li[0]):
                    a.append(torch.index_select(param[i], 0, non_zero_index))
                tensor = torch.stack(a, 0)
                non_zero_index = torch.tensor(np.arange(li[0]))
                state_dict['pred_{}'.format(cnt)] = tensor
                non_zero_index_dict['non_zero_index_{}'.format(cnt)] = non_zero_index
            elif 'batch_norm' in li_name[-2] and li_name[-1] == 'bias':
                cnt += 1
    # -------------------------------------------------------------------------------------------------#

    # cfg 생성
    # ---------------------------------------------------------------------------------------------#

    # 생성될 파일
    write_file = open(cfg_compressed, 'w')

    # 기존 cfg에서 원본 모델 구조 가져오기
    module_defs = parse_model_config(cfg_origin)

    # FPGM 적용된 모델에서 non_zero_index_dict 가져오기
    # model_dict = torch.load('./FPGM/weights/non_zero_index.pth')
    # non_zero_index_dict = model_dict['non_zero_index_dict']

    # cnt는 convolutional이나 predict일 때만 +1 된다.
    cnt = 0
    for i, module in enumerate(module_defs):
        if module['type'] == 'net':
            continue
        elif module['type'] == 'convolutional':
            # print('cnt = ' + str(cnt))
            filter = len(non_zero_index_dict['non_zero_index_{}'.format(cnt)])
            module['filters'] = filter
            non_zero_index = ','.join(list(map(str, non_zero_index_dict['non_zero_index_{}'.format(cnt)].tolist())))
            # print(non_zero_index)
            module['non_zero_index'] = non_zero_index
            cnt += 1

        elif module['type'] == 'shortcut':
            # shortcut이 연속으로 있는 경우 대비
            shortcut_layer = []
            num = 3
            while module_defs[i - num]['type'] == 'shortcut':
                shortcut_layer.append(1 + num)
                num += 3
            shortcut_layer.append(num)

            or_index = set(list(map(int, list(module_defs[i - 1]['non_zero_index'].split(',')))))
            for j in shortcut_layer:
                front_index = set(list(map(int, list(module_defs[i - j]['non_zero_index'].split(',')))))
                or_index = or_index | front_index
            or_index = list(or_index)

            filter = len(or_index)
            non_zero_index = ','.join(list(map(str, or_index)))

            module_defs[i - 1]['non_zero_index'] = non_zero_index
            module_defs[i - 1]['filters'] = filter
            for j in shortcut_layer:
                module_defs[i - j]['non_zero_index'] = non_zero_index
                module_defs[i - j]['filters'] = filter

            # import pdb; pdb.set_trace()
        elif module['type'] == 'predict':
            # FPGM 적용하지 않아서 filter를 그대로 둔다.
            # input_channel을 위해 index만 가지고 있는다.
            # print('cnt = ' + str(cnt))
            non_zero_index = ','.join(list(map(str, non_zero_index_dict['non_zero_index_{}'.format(cnt)].tolist())))
            module['non_zero_index'] = non_zero_index
            cnt += 1

        elif module['type'] == 'route':
            # 이 경우에는 [-4], [-1, 61], [-4], [-1, 36]의 경우가 있다
            # 개수가 1개와 두개로 나누어 우선 처리

            # -4인 경우
            if len(module['layers'].split(',')) == 1:
                module['non_zero_index'] = module_defs[i - 4]['non_zero_index']

            # -1, 61 또는 -1, 36인 경우
            elif len(module['layers'].split(',')) == 2:
                a, b = map(int, module['layers'].split(','))
                if b == 61:
                    non_zero_a = list(map(int, module_defs[i + a - 1]['non_zero_index'].split(',')))
                    non_zero_b = list(map(int, module_defs[b]['non_zero_index'].split(',')))

                    for i in range(len(non_zero_b)):  # 원래 256 + 512 concat
                        non_zero_b[i] += 256

                    non_zero_index = non_zero_a + non_zero_b
                    module['non_zero_index'] = ','.join(map(str, non_zero_index))

                elif b == 36:
                    non_zero_a = list(map(int, module_defs[i + a - 1]['non_zero_index'].split(',')))
                    non_zero_b = list(map(int, module_defs[b]['non_zero_index'].split(',')))

                    for i in range(len(non_zero_b)):  # 원래 128 + 256 concat
                        non_zero_b[i] += 128

                    non_zero_index = non_zero_a + non_zero_b
                    module['non_zero_index'] = ','.join(map(str, non_zero_index))

    line = ''
    for module in module_defs:
        if module['type'] == 'net':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + value + '\n'
        elif module['type'] == 'convolutional':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + str(value) + '\n'
        elif module['type'] == 'shortcut':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + str(value) + '\n'
        elif module['type'] == 'predict':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + str(value) + '\n'
                # str은 yolo 직전 75 filter에 bn이 없어서 정수 0으로 되어있기 때문
        elif module['type'] == 'yolo':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + value + '\n'
        elif module['type'] == 'route':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + value + '\n'
        elif module['type'] == 'upsample':
            line += '[' + module['type'] + ']' + '\n'
            del module['type']
            for key, value in module.items():
                line += key + '=' + value + '\n'

    write_file.write(line)
    # print(line)

    write_file.close()

    # ----------------------------------------------------------------------------------#

    # .pth 생성------------------------------------------------------------------------------#

    model_origin = Darknet(cfg_origin)
    model_origin.load_state_dict(torch.load(pretrained, map_location=device))  # FPGM 압축한 가중치

    model_compressed = Darknet(cfg_compressed)

    module_defs = parse_model_config(cfg_compressed)

    non_zero_index = ['0,1,2']
    for module in module_defs:
        # import pdb; pdb.set_trace()
        if 'non_zero_index' in module:
            non_zero_index.append(module['non_zero_index'])

    # named_parameters() 대신에 state_dict().items()를 사용하는 이유는 running_mean, running_var도 복사해야하기 때문이다.
    non_zero_index_i = 0
    for model_ori, model_comp in zip(model_origin.state_dict().items(), model_compressed.state_dict().items()):
        # print(model_ori[0], non_zero_index_i)

        # route가 중간에 4개 끼어 있는데 이때 non_zero_index_i를 보정하기 위해서 +1을 해주어 맞춰준다.
        if non_zero_index_i == 59:
            non_zero_index_i += 1  # route가 중간에 있기 때문에 +1을 해주어서 index를 맞추어야한다.
            # import pdb; pdb.set_trace()
        elif non_zero_index_i == 61:
            non_zero_index_i += 1  # route가 중간에 있기 때문에 +1을 해주어서 index를 맞추어야한다.
            # import pdb; pdb.set_trace()
        elif non_zero_index_i == 69:
            non_zero_index_i += 1  # route가 중간에 있기 때문에 +1을 해주어서 index를 맞추어야한다.
        elif non_zero_index_i == 71:
            non_zero_index_i += 1  # route가 중간에 있기 때문에 +1을 해주어서 index를 맞추어야한다.

        if len(model_ori[1].size()) == 4:
            channel_index = torch.tensor(list(map(int, non_zero_index[non_zero_index_i].split(','))))
            filter_index = torch.tensor(list(map(int, non_zero_index[non_zero_index_i + 1].split(','))))
            # import pdb; pdb.set_trace()
            temp_weight = torch.index_select(model_ori[1], 1, channel_index)
            temp_weight = torch.index_select(temp_weight, 0, filter_index)
            # import pdb; pdb.set_trace()
            model_comp[1].data.copy_(temp_weight.data)  # weight 값 깊은 복사

        elif len(model_ori[1].size()) == 1:  # bn_weight와 bn_bias
            filter_index = torch.tensor(list(map(int, non_zero_index[non_zero_index_i + 1].split(','))))
            temp_weight = torch.index_select(model_ori[1], 0, filter_index)

            model_comp[1].data.copy_(temp_weight.data)  # weight 값 깊은 복사

            # if 'bias' in model_ori[0] and not('pred_81' in model_ori[0]) and not('pred_93' in model_ori[0]) and not('pred_105' in model_ori[0]): # bn_bias인 경우
            #     non_zero_index_i += 1 # 다음 non_zero_index를 참조하기 위함
            #     # import pdb; pdb.set_trace()
        elif len(model_ori[1].size()) == 0:  # num_batches_tracked
            non_zero_index_i += 1

    # import pdb; pdb.set_trace()
    # for name, param in model_compressed.named_parameters():
    # print(name)
    # import pdb; pdb.set_trace()

    torch.save(model_compressed.state_dict(), real_compressed_pth)
    print('모델 저장 완료')


if __name__ == "__main__":
    run()