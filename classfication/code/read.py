import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import math


class LednetDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(",")
                image_name = items[0]
                label = items[1:]
                for i in range(len(label)) :
                    if label[i] == '-1.0' or math.isnan(float(label[i]))  :
                        label[i] = 0
                label = [float(i)for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
