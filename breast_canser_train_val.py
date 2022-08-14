import torch
import torch.nn as nn
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.model_selection import train_test_split
import cv2

print('————执行breast_cancer_train_val.py文件————')

root_train = 'D:/breast cancer/'
list_train_cate = os.listdir(root_train)

cate2label = {}
for i in range(len(list_train_cate)):
    if list_train_cate[i] not in cate2label:
        cate2label[list_train_cate[i]] = i

# 加载img和label
list_train_img = []
list_train_label = []
#
for cate in list_train_cate:
    list_name = os.listdir(os.path.join(root_train, cate))
    for name in list_name:
        img = cv2.imread(os.path.join(root_train, cate, name))
        img = cv2.resize(img, (256, 256))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        list_train_img.append(img)
        list_train_label.append(cate2label[cate])
print('图片个数为{}，标签个数为{}'.format(len(list_train_img), len(list_train_label)))
# 展示输入图片缩略图
# fig, axe = plt.subplots(3, 3, figsize=(256, 256))
# axe = axe.flatten()
# for i, ax in enumerate(axe):
#     ax.imshow(list_train_img[i])
#     ax.axis('off')
# plt.show()

# 单张显示图片
# img_idx = Image.fromarray(list_train_img[0])
# img_idx.show()

# 分割验证集和训练集
list_train_img, list_val_img = train_test_split(list_train_img, test_size=0.3, random_state=24)
list_train_label, list_val_label = train_test_split(list_train_label, test_size=0.3, random_state=24)

transform = transforms.Compose([
    transforms.ToTensor(),
])


class Mydata(Dataset):
    def __init__(self, x, y, transform):
        self.X = x
        self.Y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_idx = self.X[idx]
        y_idx = self.Y[idx]
        y_idx = np.array(y_idx)
        y_idx = torch.from_numpy(y_idx).type(torch.LongTensor)
        # x_idx = self.transform(Image.fromarray(x_idx).convert('RGB')).type(torch.FloatTensor)
        x_idx = self.transform(x_idx).type(torch.FloatTensor)
        return x_idx, y_idx


train_set = Mydata(list_train_img, list_train_label, transform)
val_set = Mydata(list_val_img, list_val_label, transform)

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_set,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)
print('————数据加载完毕————')
print('有{}类'.format(len(list_train_cate)))
model = models.resnet18(pretrained=True)
# for para in model.parameters():
#     para.requires_grad = False
# 冻结参数。
model.fc = nn.Linear(512, len(list_train_cate))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fun = nn.CrossEntropyLoss()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

print('————开始训练————')
loss_mean = 100
acc_epoch = 0
for epoch in range(30):
    model = model.train()
    correct = 0
    total = 0
    loss_sum = 0
    list_loss = []
    for i, (data, label) in enumerate(train_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        y_pred = model(data)
        loss = loss_fun(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        list_loss.append(loss.item())

        if i % 7 == 6:
            print('Epoch:{}|| Index:{}||Avg_Loss:{}'.format(epoch, i, loss_sum / 7))
            loss_sum = 0
    print('平均loss为{}'.format(np.mean(list_loss)))
    if np.mean(list_loss) < loss_mean:
        loss_mean = np.mean(list_loss)
        torch.save(model.state_dict(), 'D:/breast_cancer_weight/weight_best_loss.pth')

    with torch.no_grad():
        model.eval()
        for j, (data_val, label_val) in enumerate(val_loader):
            data_val = data_val.to(DEVICE)
            label_val = label_val.to(DEVICE)
            y_pred = model(data_val)
            pred_val = y_pred.max(1, keepdim=True)[1]
            total += label_val.shape[0]
            correct += pred_val.eq(label_val.view_as(pred_val)).sum().item()
        print('验证集准确率为{}'.format(correct / total))
        if correct / total > acc_epoch:
            acc_epoch = correct / total
            torch.save(model.state_dict(), 'D:/breast_cancer_weight/weight_best_acc.pth')

print('————训练结束————')
