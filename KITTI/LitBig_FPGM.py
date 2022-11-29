#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import sys
sys.path.append('/home/openlab/DH_Lee/ReconstructionPruning/')

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pytorchyolo.models import *
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
# from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary

import pandas as pd
import matplotlib.pyplot as plt



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
    parser.add_argument("-d", "--data", type=str, default="./KITTI/AISOC/AISOC.data",
                        help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=16, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, default='/home/openlab/DH_Lee/PyTorch-YOLOv3/weights/darknet53.conv.74',
                        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
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
    parser.add_argument("--seed", type=int, default=777, help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument("-g", "--gpu", type=str, default='cuda:0')
    parser.add_argument("--output-dir", type=str, default='./KITTI/train_result/LitBig_2')
    parser.add_argument("-c", type=float)
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)



    # gpu 선택 설정
    cuda_num = args.gpu
    print('gpu : {}'.format(cuda_num))
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

    # Get data configuration
    data_config = parse_data_config(args.data)
    # import pdb; pdb.set_trace()
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # ############
    # Create model
    # ############

    cfg = f'./KITTI/train_result/original_5/compress/FPGM_{args.c}/real_pruning_cfg_{args.c}.cfg'
    model = Darknet(cfg).to(device)
    model.load_state_dict(torch.load(f'./KITTI/train_result/original_5/compress/FPGM_{args.c}/weight/real_pruning_weight_{args.c}.pth'))

    # 결과 폴더 생성
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/graph", exist_ok=True)
    os.makedirs(f"{args.output_dir}/weight", exist_ok=True)

    result_path = args.output_dir

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']
    print(f'mini_batch : {mini_batch_size}')

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

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0.00001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.2)

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
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

            # Run optimizer
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()
            model.seen += imgs.size(0)

        # 수정
        # lr scheduler
        scheduler.step()

        # loss list에 추가
        li_loss.append(to_cpu(loss).item())

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
                # logger.list_of_scalars_summary(evaluation_metrics, epoch)

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
                checkpoint_path = result_path + f'/weight/yolov3_train_Litbig_{args.c}.pth'
                print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
                torch.save(model.state_dict(), checkpoint_path)
                best_MAP = AP.mean()

        # 현재 학습된 내용 DF 변환
        df = pd.DataFrame([x for x in zip(li_epoch, li_loss, li_MAP)])
        df.columns = ['epoch', 'loss', 'MAP']
        df.to_csv(result_path + f'/graph/yolov3_train_Litbig_{args.c}.csv', index=False)

        # 현재 학습된 내용 graph 변환
        plt.figure(1)
        plt.plot(li_epoch, li_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss_graph')
        plt.savefig(result_path + f'/graph/yolov3_train_Litbig_{args.c}_loss_graph.png')

        # 현재 test한 MAP graph 변환
        plt.figure(2)
        plt.plot(li_epoch, li_MAP)
        plt.xlabel('Epoch')
        plt.ylabel('MAP')
        plt.title('MAP_graph')
        plt.savefig(result_path + f'/graph/yolov3_test_Litbig_{args.c}_MAP_graph.png')


if __name__ == "__main__":
    run()
