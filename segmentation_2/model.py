from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.segmentation import fcn_resnet50


# loss
CEloss = nn.CrossEntropyLoss(reduction="sum")

# learning rate scheduler
def lr_scheduler(opt):
    return ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, verbose=1)

def optimizer(model):
    if model == fcn_resnet50:
        optimizer = optim.SGD()

# model
#     input shape : (batch_size, channel_num, width, height)
#     ouput shape : (batch_size, class_num, width, height)
fcn_resnet50_model = fcn_resnet50(pretrained=True, num_classes=21)