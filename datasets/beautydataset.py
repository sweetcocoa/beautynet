
from PIL import Image
import torch.utils.data as data
import numpy as np
from torchvision.models.resnet import *
import torch.nn as nn
import math
import torch


class BeautyDataset(data.Dataset):
    """
    data  : 파일 경로 / 점수 를 column으로 하는 pandas dataframe의 values
    """
    def __init__(self, data, transform=None, landmark_transform=None):

        self.samples = data
        self.transform = transform
        self.landmark_transform = landmark_transform

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img, landmark heatmap, target score)
        """
        path, target = self.samples[index][0], np.float32(self.samples[index][1])

        sample = np.load(path + "crop256.npy")
        landmark_heatmap = np.load(path+"heatmap.npy")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.landmark_transform is not None:
            landmark_heatmap = self.landmark_transform(landmark_heatmap)

        return sample, landmark_heatmap, target


