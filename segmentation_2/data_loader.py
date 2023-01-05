import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_tensor, to_pil_image
from albumentations import HorizontalFlip, Compose, Resize, Normalize


def transform_setting(mean, std, h, w):
    transform_train = Compose([
        Resize(h, w),
        HorizontalFlip(p=0.5),
        Normalize(mean=mean, std=std)
        ])

    transform_val = Compose([
        Resize(h, w),
        Normalize(mean=mean, std=std)
        ])
    return transform_train, transform_val


def load_dataloader(train_ds, val_ds, train_batch, val_batch):
    train_dl = DataLoader(train_ds, batch_size=train_batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=val_batch, shuffle=False)
    return train_dl, val_dl


class myVOCSegmentation(VOCSegmentation):
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            augmented = self.transforms(image=np.array(img), mask=np.array(target))
            img = augmented['image']
            target = augmented['mask']
            target[target>20] = 0

        img = to_tensor(img)
        target = torch.from_numpy(target).type(torch.long)
        return img, target