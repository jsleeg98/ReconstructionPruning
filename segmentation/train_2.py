import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='fcn_resnet50')
parser.add_argument('-tb', type=str, default='test')
parser.add_argument('-g', '--gpu', type=str, default='cuda:0')

args = parser.parse_args()

# writer=SummaryWriter(f'logs/FCN_1/{args.tb}')

# train dataset
# data augmentation
# data preprocessing
transform_train = transforms.Compose([
    transforms.PILToTensor(),
])

# test dataset
# data preprocessing
transform_test = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.voc.VOCSegmentation('../datasets/VOC_2012/', year='2012', image_set='train', download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=1,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True,
                                          num_workers=16
                                          )
import pdb; pdb.set_trace()
for data in trainloader:
    inputs, targets = data
    import pdb; pdb.set_trace()

# testset = torchvision.datasets.CIFAR10('../datasets/CIFAR10/', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset,
#                                          batch_size=128,
#                                          shuffle=False,
#                                          drop_last=True,
#                                          pin_memory=True,
#                                          num_workers=16
#                                          )

if args.model == 'fcn_resnet50':
    model = torchvision.models.segmentation.fcn_resnet50(weights=torchvision.models.segmentation.FCN_ResNet50_Weights)
    a = torch.ones(1, 3, 224, 224)
    b = model(a)
    import pdb; pdb.set_trace()
elif args.model == 'fcn_resnet101':
    model = torchvision.models.segmentation.fcn_resnet101(weights=torchvision.models.segmentation.FCN_ResNet101_Weights)


import pdb; pdb.set_trace()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.2)


device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')
model.to(device)

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
        torch.save(model.state_dict(), f'./ResNet/train_result/original/weight_2/{args.tb}.pth')
        best_acc = acc
        print('save model')
    writer.flush()

print(f'best acc : {best_acc}')
writer.add_text('best acc', str(best_acc))
writer.close()



