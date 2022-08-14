import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2.cv2 as cv2
from torchvision import models, transforms
import pandas as pd
import torch.nn as nn

print('————病灶分类test————')
test_root = '/Users/shaotianyu01/Desktop/school/11.4/test_new'
cate2label = {}
list_img = []
list_dir = []
list_label = []
list_cate = os.listdir(test_root)
list_cate = [cate for cate in list_cate if cate != '.DS_Store']
for i in range(len(list_cate)):
    if list_cate[i] not in cate2label:
        cate2label[list_cate[i]] = i
for cate in list_cate:
    list_name = os.listdir(os.path.join(test_root, cate))
    list_name = [name for name in list_name if name != '.DS_Store']
    for name in list_name:
        img_dir = os.path.join(test_root, cate, name)
        img = cv2.imread(img_dir)
        list_dir.append(img_dir)
        img = cv2.resize(img, (128, 128))
        list_img.append(img)
        list_label.append(cate2label[cate])
print('测试集共有{}张图片'.format(len(list_img)))

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


test_set = Mydata(list_img, list_label)
test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 4)
# net = Net(model)
model.load_state_dict(torch.load('/Users/shaotianyu01/Desktop/school/11.4/12.7.py_best_acc.pth'), False)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = model.to(DEVICE)

print('开始预测')
correct = 0
total = 0
list_pred = []
with torch.no_grad():
    net.eval()
    for i, (data, label) in enumerate(test_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        y_pred = net(data)
        pred = y_pred.argmax(dim=1)
        correct += pred.eq(label.view_as(pred)).sum().item()
        list_pred += pred.cpu().numpy().tolist()
        total += label.shape[0]
print('测试集准确率为:{}'.format(correct / total))

label2cate = {}
for i in range(len(list_cate)):
    if i not in label2cate:
        label2cate[i] = list_cate[i]
list_error = []

print('显示预测错误图片')
for i in range(len(list_label)):
    if list_label[i] != list_pred[i]:
        img = cv2.imread(list_dir[i])
        # cv2.imshow('True_label:{}  Pred_label:{}'.format(list_label[i], list_pred[i]), img)
        # cv2.waitKey(0)
        list_error.append([list_dir[i], label2cate[list_label[i]], label2cate[list_pred[i]]])
names = ['图片地址', '真实类别', '预测类别']

print('保存预测错误结果')
ans_df = pd.DataFrame(columns=names, data=list_error)
ans_df.to_csv('/Users/shaotianyu01/Desktop/school/12.7/error.csv')
print('运行完毕')
