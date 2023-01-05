import os
import argparse
from model import fcn_resnet50_model, optimizer, CEloss, lr_scheduler
from utils import list_check, device
from data_loader import transform_setting, myVOCSegmentation, load_dataloader
import torch

def eval(model, criterion, val_dl, device):
    model.eval()
    len_data = len(val_dl.dataset)
    val_running_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)["out"]
            loss = criterion(output, yb)
            val_running_loss += loss.item()
    val_loss = val_running_loss / float(len_data)
    return val_loss

def train(num_epochs, model, criterion, opt, lr_scheduler, train_dl, val_dl, device, model_dir):
    for epoch in range(num_epochs):
        # train
        model.to(device)
        model.train()
        running_loss=0.0
        len_data = len(train_dl.dataset)
        for xb, yb in train_dl:
            opt.zero_grad()
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            output = output['out']
            loss = criterion(output, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        loss_epoch = running_loss/float(len_data)

        #evaluation per epoch
        val_loss = eval(model, criterion, val_dl, device)

        epoch = str(epoch).zfill(len(str(num_epochs)))
        print(epoch)
        val_loss = round(val_loss, 4)
        filename = f"epoch_{epoch}_loss_{val_loss}.pt"
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, filename))
        print(f"Epoch {epoch}, loss={val_loss}")
        lr_scheduler.step(val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device option
    parser.add_argument('--device', type=str, default="cuda:0")

    # image transform
    parser.add_argument('--nomalize_mean', default=[0.485, 0.456, 0.406], type=int, nargs='+')
    parser.add_argument('--nomalize_std', default=[0.229, 0.224, 0.225], type=int, nargs='+')
    parser.add_argument('--resize', default=[520, 520], type=int, nargs='+')

    # model parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch', type=int, default=4)
    parser.add_argument('--val_batch', type=int, default=4)

    # direction
    parser.add_argument('--model_dir', type=str, default="model")
    parser.add_argument('--data_save_dir', type=str, default="../datasets/VOC_2012")

    args = parser.parse_args()

    device = device(args.device)

    # transform setting
    mean, std, h, w = list_check(args.nomalize_mean, args.nomalize_std, args.resize)
    transform_train, transform_val = transform_setting(mean, std, h, w)

    # load VOC data
    train_ds = myVOCSegmentation(
            args.data_save_dir,
            year='2012',
            image_set="train",
            download=True,
            transforms=transform_train
            )

    val_ds = myVOCSegmentation(
            args.data_save_dir,
            year='2012',
            image_set="val",
            download=True,
            transforms=transform_val
            )

    train_dl, val_dl = load_dataloader(train_ds, val_ds, args.train_batch, args.val_batch)

    # optimizer and learning rate scheduler
    opt = optimizer(deeplab_model)
    lr_scheduler = lr_scheduler(opt)

    # model train with CEloss, optm, and lr_scheduler in device
    train(args.num_epochs, deeplab_model, CEloss, opt, lr_scheduler, train_dl, val_dl, device, args.model_dir)