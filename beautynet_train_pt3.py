import torch
from skimage import io
import face_alignment
import os
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as transforms
import datasets.beautydataset as beautydataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import models.beautynet as beautynet
import random
from tqdm import tqdm


class args:
    seed = 1
    num_train = 3200
    batch_size = 32
    epochs = 25
    checkpoint = "./checkpoints/beautynet_04181047_no_person"
    pretrained = True


def set_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_valid_loss(beautynet, criterion, valid_loader):
    # beautynet.eval()
    valid_loss = 0
    accurate = 0

    for step, (img, landmark, sex, score) in enumerate(valid_loader):
        ch4img = torch.cat([img, landmark], dim=1)
        ch4img, sex, score = Variable(ch4img, volatile=True).cuda(), Variable(sex, volatile=True).cuda(), Variable(score, volatile=True).cuda()
        pred_score = beautynet(ch4img, sex)
        loss = criterion(pred_score, score)
        valid_loss += loss.cpu().data[0] * len(img)
    # accurate += (score == pred_score.max(dim=1)[1]).sum().item()
    beautynet.train(True)
    return valid_loss / (len(valid_loader.dataset) + 1), accurate / len(valid_loader.dataset)


def save_checkpoint(save_dict, path, is_best):
    torch.save(save_dict, path)
    if is_best:
        torch.save(save_dict, path+".best")


def train(net, optimizer, criterion, train_loader):
    net.train(True)
    avg_loss = 0
    for step, (img, landmarks, sex, score) in enumerate(train_loader):
        # print(img.shape, landmarks.shape, sex.shape)
        ch4img = torch.cat([img, landmarks], dim=1)
        # ch4img, sex, score = ch4img.cuda(), sex.cuda(), score.cuda()
        ch4img, sex, score = Variable(ch4img).cuda(), Variable(sex).cuda(), Variable(score).cuda()
        optimizer.zero_grad()
        net.zero_grad()
        pred_score = net(ch4img, sex)
        loss = criterion(pred_score, score)
        loss.backward()
        optimizer.step()
        avg_loss += loss.cpu().data[0] * len(ch4img)

    return avg_loss


def load_model(net : beautynet.BeautyNet, optimizer, path):
    saved_dict = torch.load(path)
    net.load_state_dict(saved_dict['net.state_dict'])
    criterion = saved_dict['criterion']
    optimizer.load_state_dict(saved_dict['optimizer.state_dict'])
    epoch = saved_dict['epoch']
    best_val_loss = saved_dict['best_val_loss']
    return net, criterion, optimizer, epoch, best_val_loss


def main():
    set_seeds(args.seed)
    dataset = pd.read_csv("./datasets/median.csv")
    dataset = dataset.sample(frac=1.0)
    train_dataset = dataset.iloc[:args.num_train]
    valid_dataset = dataset.iloc[args.num_train:]

    transform = transforms.Compose([
        # transforms.RandomResizedCrop(350, scale=(0.5, 1.0), ratio=(1., 1.)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        torch.FloatTensor,
        transforms.Normalize(mean=[174.5856, 152.9040, 145.1345], std=[82.1571, 84.6071,  87.3680]),   # RGB mean, std.
    ])

    landmark_transform = torch.FloatTensor  # None

    dset_train = beautydataset.BeautyDataset(train_dataset.values, transform=transform,
                                             landmark_transform=landmark_transform, loader=np.load)

    transform_valid = transforms.Compose([
        # transforms.Resize(350),
        # transforms.ToTensor(),
        torch.FloatTensor,
        transforms.Normalize(mean=[174.5856, 152.9040, 145.1345], std=[82.1571, 84.6071, 87.3680]),
    ])

    dset_valid = beautydataset.BeautyDataset(valid_dataset.values, transform=transform_valid,
                                             landmark_transform=landmark_transform, loader=np.load)

    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                               num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dset_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                               num_workers=4)

    net = beautynet.BeautyNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=60).cuda()

    optimizer = optim.Adam(net.parameters())
    criterion = nn.MSELoss()

    if args.pretrained:
        net, criterion, optimizer, start_epoch, best_val_loss = load_model(net, optimizer, args.checkpoint)
    else:
        best_val_loss = 100
        start_epoch = 0

    net.train(True)

    for epoch in tqdm(range(start_epoch + 1, start_epoch + args.epochs + 1)):
        avg_loss = train(net, optimizer, criterion, train_loader)
        if epoch % 1 == 0:
            val_loss, accuracy = get_valid_loss(net, criterion, valid_loader)
            is_best = val_loss < best_val_loss
            best_val_loss = val_loss if is_best else best_val_loss
            save_dict = dict(
                {'net.state_dict': net.state_dict(),
                 'criterion': criterion,
                 'optimizer.state_dict': optimizer.state_dict(),
                 'epoch': epoch,
                 'best_val_loss': best_val_loss}
            )
            save_checkpoint(save_dict, args.checkpoint, is_best)
            print(
                f"epoch{epoch}, loss{avg_loss/(len(train_loader.dataset) + 1)}, val_loss{val_loss}, best_val_loss{best_val_loss} accuracy{accuracy}")

    net.eval()

    pred_list = []
    for step, (img, landmark, sex, score) in enumerate(valid_loader):
        ch4img = torch.cat([img, landmark], dim=1)
        ch4img, sex, score = Variable(ch4img, volatile=True).cuda(), Variable(sex, volatile=True).cuda(), Variable(score, volatile=True).cuda()
        pred_score = net(ch4img, sex)
        for i in range(len(img)):
            pred_list.append(pred_score[i].cpu().data[0])
            if abs(score[i].cpu().data[0] - pred_score[i].cpu().data[0]) < 0.2:
                res = True
            else:
                res = False
            print(f"valid true score{score[i].cpu().data[0]:.4f}, pred score{pred_score[i].cpu().data[0]:.4f} result {res}" )

    result_valid = pd.Series(data=pred_list)

    pred_list = []
    for step, (img, landmark, sex, score) in enumerate(train_loader):
        ch4img = torch.cat([img, landmark], dim=1)
        ch4img, sex, score = Variable(ch4img, volatile=True).cuda(), Variable(sex, volatile=True).cuda(), Variable(score, volatile=True).cuda()
        pred_score = net(ch4img, sex)
        for i in range(len(img)):
            pred_list.append(pred_score[i].cpu().data[0])
            if abs(score[i].cpu().data[0] - pred_score[i].cpu().data[0]) < 0.2:
                res = True
            else:
                res = False
                print(
                    f"valid true score{score[i].cpu().data[0]:.4f}, pred score{pred_score[i].cpu().data[0]:.4f} result {res}")

    result_test = pd.Series(data=pred_list)
    true_valid = valid_dataset['Rating']
    true_train = train_dataset['Rating']
    import matplotlib.pyplot as plt

    plt.hist([result_valid, result_test, true_valid, true_train], color=['blue', 'green', 'orange', 'red'], bins=10, normed=True)
    plt.savefig(args.checkpoint + ".result.png")


if __name__ == "__main__":
    main()