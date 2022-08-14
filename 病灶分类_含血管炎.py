import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import cv2.cv2 as cv2


print('————12.7 执行病灶分类.py文件————')
root_train = '/Users/shaotianyu01/Desktop/school/11.4/cut_new'
list_train_cate = os.listdir(root_train)
cate2label = {}
for i in range(len(list_train_cate)):
    if list_train_cate[i] not in cate2label:
        cate2label[list_train_cate[i]] = i

# 加载img和label
list_train_img = []
list_train_label = []
for cate in list_train_cate:
    list_name = os.listdir(os.path.join(root_train, cate))
    for name in list_name:
        img = cv2.imread(os.path.join(root_train, cate, name))
        img = cv2.resize(img, (128, 128))
        # cv2.imshow('img', img)
        # cv2.waitKey(1)
        list_train_img.append(img)
        list_train_label.append(cate2label[cate])
print('图片个数为{}，标签个数为{}'.format(len(list_train_img), len(list_train_label)))

root_val = '/Users/shaotianyu01/Desktop/school/11.4/test_new'

list_val_cate = os.listdir(root_val)
list_val_img = []
list_val_label = []
for cate in list_val_cate:
    list_name_val = os.listdir(os.path.join(root_val, cate))
    for name in list_name_val:
        img = cv2.imread(os.path.join(root_val, cate, name))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (128, 128))
        list_val_img.append(img)
        list_val_label.append(cate2label[cate])


class Mydata(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.as_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.625, 0.448, 0.688], [0.131, 0.177, 0.101]),
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_idx = self.X[idx]
        y_idx = self.Y[idx]
        x_idx = self.as_tensor(x_idx)
        y_idx = np.array(y_idx)
        y_idx = torch.from_numpy(y_idx).type(torch.LongTensor)
        x_idx = x_idx.type(torch.FloatTensor)
        return x_idx, y_idx


train_set = Mydata(list_train_img, list_train_label)
val_set = Mydata(list_val_img, list_val_label)

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_set,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

print('————数据加载完毕————')

# model = Net(models.resnet18(pretrained=True))
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fun = nn.CrossEntropyLoss()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

print('————开始训练————')
loss_mean = 100
acc_epoch = 0
for epoch in range(5):
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

        if i % 1 == 0:
            print('Epoch:{}|| Index:{}||Avg_Loss:{}'.format(epoch, i, loss_sum / 1))
            loss_sum = 0
    print('平均loss为{}'.format(np.mean(list_loss)))
    if np.mean(list_loss) < loss_mean:
        loss_mean = np.mean(list_loss)
        torch.save(model.state_dict(), '/Users/shaotianyu01/Desktop/school/11.4/12.7.py_best_loss.pth')

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
            torch.save(model.state_dict(), '/Users/shaotianyu01/Desktop/school/11.4/12.7.py_best_acc.pth')

print('————训练结束————')
