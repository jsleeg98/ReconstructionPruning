import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter('logs/ResNet/')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100('../datasets/CIFAR100/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100('../datasets/CIFAR100/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  # ImageNet pretrained model
model.fc.out_features = 100  # CIFAR100을 위한 model 바꿈

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min=0.0001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        torch.save(model.state_dict(), './ResNet/train_result/original/weight/resnet50_CIFAR100_best.pth')
        print('save model')
    writer.flush()

writer.close()



