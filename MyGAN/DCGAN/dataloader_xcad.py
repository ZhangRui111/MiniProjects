import glob
import os
from torch.utils.data import Dataset
from PIL import Image


class XCADImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = "{}/XCAD".format(img_dir)
        self.img_list = self.images_list()
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('L')
        label = 1  # 1 means real/valid image
        if self.transform:
            image = self.transform(image)
        return image, label

    def images_list(self):
        imgs = glob.glob("{}/*".format(self.img_dir))
        imgs = [i.replace('\\', '/') for i in imgs]  # for WinOS
        return imgs
