import torch
import face_alignment
import os
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as transforms
import models.beautynet as beautynet
from tqdm import tqdm
from utils import image_preprocess
import glob
import torchvision.datasets.folder as folder


class args:
    seed = 1
    num_train = 3200
    batch_size = 1
    epochs = 32
    checkpoint = "./checkpoints/beautynet_04120917_4way"
    pretrained = False


def load_model(net : beautynet.BeautyNet, path : str):
    saved_dict = torch.load(path)
    net.load_state_dict(saved_dict['net.state_dict'])
    net.eval()
    return net


def inference_image(fa, net, path):
    img = folder.default_loader(path)

    img_crop = image_preprocess.get_cropped_sample(fa, np.array(img))
    if img_crop is None:
        return 0

    preds = fa.get_landmarks(img_crop)[0]
    lanimg = image_preprocess.get_landmark_to_img(preds)
    transform = transforms.Compose([
        torch.FloatTensor,
        transforms.Normalize(mean=[174.5856, 152.9040, 145.1345], std=[82.1571, 84.6071,  87.3680]),   # RGB mean, std.
    ])

    img_crop_t = img_crop.transpose([2,0,1])
    img_crop_t = transform(img_crop_t)

    lanimg = torch.FloatTensor(lanimg.astype(np.float32))

    # print(img_crop_t.shape, lanimg.shape)
    ch4img = torch.cat([img_crop_t, lanimg], dim=0).unsqueeze(0)

    personal = torch.FloatTensor([0, 0, 0, 0]).unsqueeze(0)

    with torch.no_grad():
        ch4img = ch4img.cuda()
        personal = personal.cuda()
        score = net(ch4img, personal)

    return score.item()


def get_bottleneck_of_image(fa, net, path, layer=0):
    """
    :param fa:
    :param net:
    :param path: list of file paths or single file path(str)
    :return: bottleneck tensor
    """
    if isinstance(path, list):
        img_paths = path
    elif isinstance(path, str):
        img_paths = [path]

    cropped_ch4img = []
    for i, img_path in enumerate(img_paths):
        img = folder.default_loader(img_path)
        img_crop = image_preprocess.get_cropped_sample(fa, np.array(img))
        if img_crop is None:
            return None
        preds = fa.get_landmarks(img_crop)[0]
        lanimg = image_preprocess.get_landmark_to_img(preds)

        mean = [174.5856, 152.9040, 145.1345]
        std = [82.1571, 84.6071, 87.3680]

        img_crop = img_crop.transpose([2, 0, 1])
        for channel in range(len(img_crop)):

            img_crop[channel] = (img_crop[channel] - mean[channel]) / std[channel]

        lanimg = lanimg.astype(np.float32)
        ch4img = np.concatenate((img_crop, lanimg), axis=0)
        cropped_ch4img.append(ch4img)

    cropped_ch4img = torch.FloatTensor(cropped_ch4img)
    with torch.no_grad():
        personal = torch.zeros(len(cropped_ch4img), 4).cuda()
        cropped_ch4img = cropped_ch4img.cuda()
        bottleneck = net.get_bottleneck(cropped_ch4img, personal, layer=1)

    return bottleneck.cpu().data


def main():
    """
        Module의 Unit Test Code
        테스트 데이터들에 대한 점수 분포 / 실제 정답 데이터에 대한 점수 분포 비교
    """
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, enable_cuda=True)
    net = beautynet.BeautyNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=60).cuda()
    net = load_model(net, args.checkpoint)

    pred_list = []
    image_list = glob.glob(os.path.join("./datasets/ImageFolder/", "fty*.jpg"))
    for i, path in tqdm(enumerate(sorted(image_list))):
        score = inference_image(fa, net, path)
        pred_list.append(score)
        if i % 20 == 0:
            print(score)

    result_fty = pd.Series(data=pred_list)

    true_data = pd.read_csv("./datasets/median.csv")
    true_fty = true_data.loc[true_data['Female'] == 1, "Rating"]
    import matplotlib.pyplot as plt

    plt.hist([result_fty, true_fty], color=['blue', 'red'], bins=10, normed=True)
    plt.savefig(args.checkpoint + ".inf_result.png")
    plt.show()


if __name__ == "__main__":
    main()