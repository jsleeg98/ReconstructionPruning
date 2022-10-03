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
        self.union_index = []

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

    def union_remain_index(self):
        first = self.remain_index[0]  # resnet34 first conv1 insert to layer1
        self.remain_index[1].insert(0, [first])
        del self.remain_index[0]

        for li_li_remain_index in self.remain_index:
            set_union_index = set()
            for li_remain_index in li_li_remain_index:
                if len(li_remain_index) != 3:
                    set_remain_index = set(li_remain_index[-1])
                    set_union_index = set_union_index.union(set_remain_index)
                elif len(li_remain_index) == 3:
                    set_remain_index = set(li_remain_index[1])
                    set_union_index = set_union_index.union(set_remain_index)
                    set_remain_index = set(li_remain_index[2])
                    set_union_index = set_union_index.union(set_remain_index)
            self.union_index.append(list(set_union_index))

