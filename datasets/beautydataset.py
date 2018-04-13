# from PIL import Image
# import torch.utils.data as data
# import torchvision.datasets.folder as folder
# import numpy as np
# from torchvision.models.resnet import *
# import torch.nn as nn
# import math
#
#
#
#
#
# class BeautyDataset(data.Dataset):
#     """
#
#     """
#     def __init__(self, data, transform=None, landmark_transform=None, loader=folder.default_loader):
#         self.samples = data
#         self.loader = loader
#         self.transform = transform
#         self.landmark_transform = landmark_transform
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (img, landmarks, sex, score)
#         """
#         path, target, sex = self.samples[index][0], np.float32(self.samples[index][1]), np.float32(self.samples[index][2]) # , self.samples[index][2:].astype(np.int32)
#
#
#         sample = self.loader(path)
#         landmark_image = np.load(f"{path[:path.rfind('.')]}.landmark.npy").astype(np.int32)
#         # print(f"path{path}")
#         if self.transform is not None:
#             sample = self.transform(sample)
#
#
#         if self.landmark_transform is not None:
#             landmark_image = self.landmark_transform(landmark_image)
#
#         return sample, landmark_image, sex, target


from PIL import Image
import torch.utils.data as data
import torchvision.datasets.folder as folder
import numpy as np
from torchvision.models.resnet import *
import torch.nn as nn
import math
import face_alignment
from face_alignment.utils import crop
import torch


class BeautyDataset(data.Dataset):
    """

    """

    def __init__(self, data, transform=None, landmark_transform=None, loader=folder.default_loader, fa=None):
        self.samples = data
        self.loader = loader
        self.transform = transform
        self.landmark_transform = landmark_transform
        self.fa = fa

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_cropped_sample(self, sample):
        d = self.fa.face_detector(sample).pop().rect
        center = torch.FloatTensor(
            [d.right() - (d.right() - d.left()) / 2.0, d.bottom() - (d.bottom() - d.top()) / 2.0])
        center[1] = center[1] - (d.bottom() - d.top()) * 0.12
        scale = (d.right() - d.left() + d.bottom() - d.top()) / 224.0
        img_crop = crop(sample, center, scale, resolution=350)
        return img_crop

    def get_landmarks(self, img):
        return self.fa.get_landmarks(img)[0]

    @staticmethod
    def get_landmark_to_img(landmark):
        lanimg = np.zeros((1, 350, 350)).astype(np.int8)
        for x, y in landmark:
            x, y = x.astype(np.int), y.astype(np.int)
            x = np.clip(x, 0, 349)
            y = np.clip(y, 0, 349)
            lanimg[0, y, x] = 1
        return lanimg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img, landmarks, personality, score)
        """
        path, target, personality = self.samples[index][0], np.float32(self.samples[index][1]), np.float32(self.samples[index][2:6]) # , self.samples[index][2:].astype(np.int32)
        personality = np.float32([0, 0, 0, 0])
        sample = self.loader(path + "_cimg.npy")
        landmark_image = np.load(path+"_ldmk.npy")

        # sample = self.get_cropped_sample(sample)
        # landmark_image = np.load(f"{path[:path.rfind('.')]}.landmark.npy").astype(np.int32)
        # landmark = self.get_landmarks(sample)
        # landmark_image = BeautyDataset.get_landmark_to_img(landmark).astype(np.int32)

        # sample = sample.transpose([1,0,2])
        sample = np.float32(sample)
        landmark_image = np.float32(landmark_image)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.landmark_transform is not None:
            landmark_image = self.landmark_transform(landmark_image)

        return sample, landmark_image, personality, target

