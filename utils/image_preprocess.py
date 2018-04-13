import face_alignment
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


def get_cropped_sample(fa, sample):
    """
    :param fa: face_alignment
    :param sample: 3-channel numpy array image, Size > 224 x 224.
    :return: 350 x 350 x 3 numpy image
    """
    d = fa.face_detector(sample)
    if len(d) == 0:
        return None
    d = d.pop().rect

    center = torch.FloatTensor(
        [d.right() - (d.right() - d.left()) / 2.0, d.bottom() - (d.bottom() - d.top()) / 2.0])
    center[1] = center[1] - (d.bottom() - d.top()) * 0.12
    scale = (d.right() - d.left() + d.bottom() - d.top()) / 224.0
    img_crop = crop(sample, center, scale, resolution=350)
    return img_crop


def get_landmarks(fa, img):
    """
    :param fa: face_alignment
    :param img: 350 x 350 x 3 numpy image
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



