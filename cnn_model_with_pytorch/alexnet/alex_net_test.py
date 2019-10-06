# coding=utf-8

import torch
import cv2
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from cnn_model_with_pytorch.alexnet.alex_net import AlexNet
from tensorboardX import SummaryWriter
import numpy as np
import os

root = "./alexnet/"
path = '/home/zhongjianping/桌面/mynet/cnn_model_with_pytorch/alexnet/dataset/'
writer = SummaryWriter('./log')


# -----------------ready the dataset--------------------------
def opencvLoad(imgPath, resizeH, resizeW):
    image = cv2.imread(path+imgPath)
    image = cv2.resize(image, (resizeH, resizeW), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))
    image = torch.from_numpy(image)
    return image


class LoadPartDataset(Dataset):
    def __init__(self, txt):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            labelList = int(words[1])
            imageList = words[0]
            imgs.append((imageList, labelList))
        self.imgs = imgs

    def __getitem__(self, item):
        image, label = self.imgs[item]
        img = opencvLoad(image, 224, 224)
        return img, label

    def __len__(self):
        return len(self.imgs)


def loadTrainData(txt=None):
    fh = open(txt, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        print(words)
        label = int(words[1])
        image = cv2.imread(words[0])

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 1, 0))
        image = torch.from_numpy(image)
        imgs.append((image, label))
    return imgs


# trainSet=loadTrainData(txt=root+'train.txt')
# test_data=loadTrainData(txt=root+'train.txt')
trainSet = LoadPartDataset(txt=root + 'train.txt')
test_data = LoadPartDataset(txt=root + 'test.txt')
train_loader = DataLoader(dataset=trainSet, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)

device = torch.device("cuda")
model = AlexNet()
lr = 1e-5
loss_func = torch.nn.CrossEntropyLoss()
torch.optim.Adam()
optimizer = torch.optim.SGD(list(model.parameters())[:], lr=lr, momentum=0.9)
PATH = "/home/zhongjianping/桌面/mynet/cnn_model_with_pytorch/alexnet/model/_iter_999.pth"
if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH))
model.to(device)

for epoch in range(2000):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for i, data in enumerate(train_loader, 0):
        trainData, trainLabel = data
        trainData, trainLabel = Variable(trainData.cuda()), Variable(trainLabel.cuda())
        out = model(trainData)
        loss = loss_func(out, trainLabel)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == trainLabel).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            number_iter = epoch*len(train_loader) + i
            writer.add_scalar('Train/Loss', loss.item(), number_iter)
            writer.add_scalar('Train/Acc', train_correct.double()/len(trainLabel), number_iter)
    #  if epoch % 100 == 0:
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        trainSet)), train_acc / (len(trainSet))))

    if (epoch + 1) % 100 == 0:
        sodir = '/home/zhongjianping/桌面/mynet/cnn_model_with_pytorch/alexnet/model/_iter_{}.pth'.format(epoch)
        print('[5] Model save {}'.format(sodir))
        if not os.path.exists(sodir):
            f = open(sodir, 'w')
            f.close()
        torch.save(model.state_dict(), sodir)

    # adjust
    if (epoch + 1) % 100 == 0:
        lr = lr / 10
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# evaluation--------------------------------
model.eval()
eval_loss = 0.
eval_acc = 0.
for trainData, trainLabel in test_loader:
    trainData, trainLabel = Variable(trainData.cuda()), Variable(trainLabel.cuda())
    out = model(trainData)
    loss = loss_func(out, trainLabel)
    eval_loss += loss.item()
    pred = torch.max(out, 1)[1]
    num_correct = (pred == trainLabel).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_data)), eval_acc / (len(test_data))))
