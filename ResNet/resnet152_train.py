import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models import resnet
writer=SummaryWriter('logs/ResNet/resnet152_8/')

# train dataset
# data augmentation
# data preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
])

# test dataset
# data preprocessing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
])

trainset = torchvision.datasets.CIFAR100('../datasets/CIFAR100/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR100('../datasets/CIFAR100/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

model = resnet.resnet152(num_classes=100)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')
model.to(device)

best_acc = 0
for epoch in range(200):  # loop over the dataset multiple times
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
        torch.save(model.state_dict(), './ResNet/train_result/original/weight/resnet152_CIFAR100_best_8.pth')
        best_acc = acc
        print('save model')
    writer.flush()

print(f'best acc : {best_acc}')
writer.close()



