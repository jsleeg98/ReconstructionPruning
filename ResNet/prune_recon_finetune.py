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
import PruneCustom
import PruneHandler as PH
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='resnet34')
parser.add_argument('-tb', type=str, default='test')
parser.add_argument('-c', '--compression_ratio', type=float, default=0.5)
parser.add_argument('-g', '--gpu', type=str, default='cuda:0')
parser.add_argument('-ln', type=int, default=-1)
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-gamma', type=float, default=0.2)
parser.add_argument('-wd', type=float, default=0.0001)

args = parser.parse_args()

writer=SummaryWriter(f'logs/ResNet_6/{args.tb}_{args.compression_ratio}_{args.ln}')

for name, value in vars(args).items():
    print(f'{name} : {value}')
    writer.add_text(f'{name}', f'{value}')

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
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True,
                                          num_workers=16
                                          )

testset = torchvision.datasets.CIFAR10('../datasets/CIFAR10/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=128,
                                         shuffle=False,
                                         drop_last=True,
                                         pin_memory=True,
                                         num_workers=16
                                         )

if args.model == 'resnet34':
    model = torchvision.models.resnet34(pretrained=False)
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.conv1 = conv1
    model.fc.out_features = 10
    model.load_state_dict(torch.load('./ResNet/train_result/original/weight_3/resnet34_1.pth'))
elif args.model == 'resnet50':
    model = torchvision.models.resnet50(pretrained=False)
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.conv1 = conv1
    model.fc.out_features = 10
    model.load_state_dict(torch.load('./ResNet/train_result/original/weight_3/resnet50_1.pth', map_location='cpu'))
elif args.model == 'resnet101':
    model = torchvision.models.resnet101(pretrained=False)
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.conv1 = conv1
    model.fc.out_features = 10
    model.load_state_dict(torch.load('./ResNet/train_result/original/weight_3/resnet101_1.pth'))

if args.ln == 1 or args.ln == 2:
    print(f'norm {args.ln} magnitude')
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=args.compression_ratio, n=args.ln, dim=0)
            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
        if isinstance(module, torch.nn.BatchNorm2d):
            prune.l1_unstructured(module, name='weight', amount=args.compression_ratio, importance_scores=mask)
            prune.l1_unstructured(module, name='bias', amount=args.compression_ratio, importance_scores=mask)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
        if isinstance(module, torch.nn.BatchNorm2d):
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')
elif args.ln == -1:
    print('FPGM')
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            PruneCustom.gm_structured(module, name='weight', amount=args.compression_ratio, dim=0)
            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
        if isinstance(module, torch.nn.BatchNorm2d):
            prune.l1_unstructured(module, name='weight', amount=args.compression_ratio, importance_scores=mask)
            prune.l1_unstructured(module, name='bias', amount=args.compression_ratio, importance_scores=mask)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
        if isinstance(module, torch.nn.BatchNorm2d):
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')


macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

ph = PH.PruneHandler(model)
if args.model in ['resnet18', 'resnet34']:
    model = ph.reconstruction_model('basic')
elif args.model in ['resnet50', 'resnet101', 'resnet152']:
    model = ph.reconstruction_model('bottle')
print('reconstruction done')

macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
writer.add_text('MAC', str(macs))
writer.add_text('PARAMs', str(params))


device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')
model.to(device)
model = nn.DataParallel(model)  # multi-GPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=args.gamma)

best_acc = 0
for epoch in range(100):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    print(f'epoch : {epoch + 1}')
    writer.add_scalar("LR/train", optimizer.param_groups[0]['lr'], epoch)
    for data in tqdm(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss
        loss.backward()
        optimizer.step()
    scheduler.step()
    writer.add_scalar("Loss/train", running_loss, epoch)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Accuracy : {100 * correct / total}%')
    writer.add_scalar("Acc/test", acc, epoch)

    if best_acc < acc:
        best_acc = acc
        torch.save(model.module.state_dict(), f'./ResNet/train_result/original/weight_3/recon/{args.tb}_{args.compression_ratio}_{args.ln}.pth')
        print('best model save')
    writer.flush()

print(f'best acc : {best_acc}')
writer.add_text('best acc', str(best_acc))
writer.close()

