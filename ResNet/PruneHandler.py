import torch
import torchvision
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models import resnet
import argparse
import copy

class PruneHandler():
    def __init__(self, model):
        self.model = model
        self.remain_index = []

    def get_remain_index(self):
        for name, module in self.model.named_children():
            if isinstance(module, torch.nn.Conv2d):
                tmp_remain_index = torch.where(torch.norm(module.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                self.remain_index.append(tmp_remain_index)
            elif isinstance(module, torch.nn.Sequential):
                li_li_remain_index = []
                for name_, module_ in module.named_children():
                    if isinstance(module_, resnet.BasicBlock):
                        li_remain_index = []
                        for name__, module__ in module_.named_children():
                            if isinstance(module__, torch.nn.Conv2d):
                                tmp_remain_index = torch.where(torch.norm(module__.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                                li_remain_index.append(tmp_remain_index)
                            elif isinstance(module__, torch.nn.Sequential):
                                for name___, module___ in module__.named_children():
                                    if isinstance(module___, torch.nn.Conv2d):
                                        tmp_remain_index = torch.where(torch.norm(module___.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                                        li_remain_index.append(tmp_remain_index)
                        li_li_remain_index.append(li_remain_index)
                self.remain_index.append(li_li_remain_index)


