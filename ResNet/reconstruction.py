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
import PruneHandler as PH
import time

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='cuda:0')
args = parser.parse_args()

# train dataset
# data augmentation
# data preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

# test dataset
# data preprocessing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10('../datasets/CIFAR10/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10('../datasets/CIFAR10/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

model = resnet.resnet34()
model.load_state_dict(torch.load('./ResNet/train_result/original/weight/resnet152_test.pth'))

device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')
# model_compressed = copy.deepcopy(model)

model.to(device)
model.eval()
correct = 0
total = 0
start = time.time()
with torch.no_grad():
    for data in tqdm(testloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
end = time.time()
print(f'걸린 시간 : {end - start}')
acc = 100 * correct / total
print(f'Accuracy : {100 * correct / total}%')

model.to('cpu')

ph = PH.PruneHandler(model)
model = ph.reconstruction_model()
print('reconstruction done')


# for name, parameter in model.named_parameters():
#     print(name)

model.to(device)
model.eval()
correct = 0
total = 0
start = time.time()
with torch.no_grad():
    for data in tqdm(testloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
end = time.time()
print(f'걸린 시간 : {end - start}')
acc = 100 * correct / total
print(f'Accuracy : {100 * correct / total}%')



