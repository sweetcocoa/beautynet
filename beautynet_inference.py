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
import matplotlib.pyplot as plt
from torch.autograd import Variable

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
    if isinstance(path, str):
        img = folder.default_loader(path)
    else:
        img = path

    img_crop = image_preprocess.get_cropped_sample(fa, np.array(img))
    if img_crop is None:
        return 0

    inp = torch.tensor(img_crop.transpose(2,0,1)).float().div(255.0).unsqueeze(0)

    if fa.enable_cuda:
        inp = inp.cuda()

    landmark_heatmap = fa.face_alignemnt_net(inp)[-1].data
    score = net(inp, landmark_heatmap)

    return score.item()


def get_bottleneck_of_image(fa, net, path, layer=0, preprocessed=True):
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
    # with torch.no_grad():
    #     personal = torch.zeros(len(cropped_ch4img), 4).cuda()
    #     cropped_ch4img = cropped_ch4img.cuda()
    #     bottleneck = net.get_bottleneck(cropped_ch4img, personal, layer=layer)

    personal = Variable(torch.zeros(len(cropped_ch4img), 4), volatile=True).cuda()
    cropped_ch4img = Variable(cropped_ch4img, volatile=True).cuda()
    bottleneck = net.get_bottleneck(cropped_ch4img, personal, layer=layer)


    return bottleneck.cpu().data


def show_img_and_landmarks(fa, img):
    """
    numpy image array -> landmark + image
    return landmark (68 x 2)
    """
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(img)
    ax = fig.add_subplot(122)
    ax.imshow(img)
    markersize = 1
    preds = fa.get_landmarks(img)[0]
    # 턱선
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=markersize,linestyle='-',color='w',lw=2)
    # 오른눈썹
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=markersize,linestyle='-',color='w',lw=2)
    # 왼눈썹
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=markersize,linestyle='-',color='w',lw=2)
    # 콧대
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=markersize,linestyle='-',color='w',lw=2)
    # 콧구멍라인
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=markersize,linestyle='-',color='w',lw=2)
    # 오른눈
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=markersize,linestyle='-',color='w',lw=1)
    # 왼눈
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=markersize,linestyle='-',color='w',lw=2)
    # 입술 밖
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=markersize,linestyle='-',color='w',lw=2)
    # 입술 안
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=markersize,linestyle='-',color='w',lw=2)
    ax.axis('off')
    return preds


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