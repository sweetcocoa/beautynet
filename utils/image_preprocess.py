import face_alignment
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.datasets.folder as folder
import numpy as np
from torchvision.models.resnet import *
import torch.nn as nn
import math
import face_alignment
from face_alignment.utils import crop
import torch
import random


def RandomEnhanceBrightness(img):
    factor = random.gauss(1, 0.2)
    factor = np.clip(factor, 0.7, 1.3)
    return ImageEnhance.Brightness(img).enhance(factor)


def RandomEnhanceColor(img):
    factor = random.gauss(1, 0.2)
    factor = np.clip(factor, 0., 1.5)
    return ImageEnhance.Color(img).enhance(factor)


def RandomEnhanceContrast(img):
    factor = random.gauss(1, 0.2)
    factor = np.clip(factor, 0.8, 2)
    return ImageEnhance.Contrast(img).enhance(factor)


def RandomEnhanceSharpness(img):
    factor = random.gauss(1, 0.3)
    factor = np.clip(factor, -1, 5)
    return ImageEnhance.Sharpness(img).enhance(factor)


def get_center_scale_from_rectangle(d):
    center = torch.FloatTensor(
        [d[2] - (d[2] - d[0]) / 2.0,
         d[3] - (d[3] - d[1]) / 2.0])
    center[1] = center[1] - (d[3] - d[1]) * 0.12
    scale = (d[2] - d[0] + d[3] - d[1]) / 195.0
    return center, scale


def get_cropped_sample(fa, sample):
    """
    :param fa: face_alignment
    :param sample: 3-channel numpy array image, Size > 195 x 195.
    :return: 256 x 256 x 3 numpy image
    """
    d = fa.face_detector(sample)

    if len(d) == 0:
        return None

    if fa.enable_cuda:
        d = d.pop().rect
    else:
        d = d.pop()

    center = torch.FloatTensor(
        [d.right() - (d.right() - d.left()) / 2.0, d.bottom() - (d.bottom() - d.top()) / 2.0])
    center[1] = center[1] - (d.bottom() - d.top()) * 0.12
    scale = (d.right() - d.left() + d.bottom() - d.top()) / 195.0
    img_crop = crop(sample, center, scale, resolution=256)
    return img_crop


def get_landmarks(fa, img):
    """
    :param fa: face_alignment
    :return: 68x2 2d landmark coord.
    """
    ret = fa.get_landmarks(img)
    if len(ret) > 0:
        return ret[0]
    else:
        return None


def get_landmark_to_img(landmark):
    """
    :param landmark: 68x2 landmark
    :return: 1x350x350 landmark image
    """
    lanimg = np.zeros((1, 350, 350)).astype(np.int8)
    for x, y in landmark:
        x, y = x.astype(np.int), y.astype(np.int)
        x = np.clip(x, 0, 349)
        y = np.clip(y, 0, 349)
        lanimg[0, y, x] = 1
    return lanimg


def save_images_state(path):
    """
    read image file, to get a cropped image and a landmark image
    :param path:
    :return:
    save npy file
    """
    img = folder.default_loader(path)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, enable_cuda=True)
    img_crop = get_cropped_sample(fa, np.array(img))
    landmark = get_landmarks(fa, img_crop)
    lanimg = get_landmark_to_img(landmark)
    np.save(path+'_ldmk', lanimg)
    img_crop_t = img_crop.transpose([2, 0, 1])
    np.save(path+'_crop', img_crop_t)



