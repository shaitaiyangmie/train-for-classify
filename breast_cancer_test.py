import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

root = 'D:/breast cancer'
list_cate = os.listdir(root)
cate2label = {}
for i in range(len(list_cate)):
    if list_cate[i] not in cate2label:
        cate2label[list_cate[i]] = i
list_img = []
list_label = []
list_dir = []
for cate in list_cate:
    list_name = os.listdir(os.path.join(root, cate))
    for name in list_name:
        img = cv2.imread(os.path.join(root, cate, name))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (256, 256))
        list_img.append(img)
        list_label.append(cate2label[cate])
        list_dir.append(os.path.join(root, cate, name))
x_train, x_val = train_test_split(list_img, test_size=0.3, random_state=24)
y_train, y_val = train_test_split(list_label, test_size=0.3, random_state=24)
dir_train, dir_val = train_test_split(list_dir, test_size=0.3, random_state=24)

# for i in range(len(list_img)):
#     cv2.imshow('{}'.format(list_label[i]), list_img[i])
#     cv2.waitKey(1)

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
        return transform(x_idx).type(torch.FloatTensor), torch.from_numpy(y_idx).type(torch.LongTensor)


test_set = Mydata(x_val, y_val, transform)
test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, len(list_cate))
model.load_state_dict(torch.load('D:/breast_cancer_weight/weight_best_acc.pth'), False)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

correct = 0
total = 0
list_pred = []
print('————开始预测————')
with torch.no_grad():
    model.eval()
    for i, (data, label) in enumerate(test_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        y_pred = model(data)
        # pred = y_pred.max(1, keepdim=True)[1]
        pred = y_pred.argmax(dim=1)
        correct += pred.eq(label.view_as(pred)).sum().item()
        list_pred += pred.cpu().numpy().tolist()
        total += label.shape[0]

print('测试准确率为：{}'.format(correct / total))
# print('预测结果为：{}'.format(list_pred))

print('————显示分类错误图片————')
label2cate = {}
for i in range(len(list_cate)):
    if i not in label2cate:
        label2cate[i] = list_cate[i]
ans = []
for i in range(len(y_val)):
    if y_val[i] != list_pred[i]: 
        ans.append([dir_val[i], label2cate[y_val[i]], label2cate[list_pred[i]]])
        img = cv2.imread(dir_val[i])
        cv2.imshow('True Label:{}, Pred Label:{}'.format(y_val[i], list_pred[i]), img)
        cv2.waitKey(0)

ans_name = ['预测错误图片路径', '真实标签', '预测结果']
df_ans = pd.DataFrame(columns=ans_name, data=ans)
df_ans.to_csv('D:/breast_cancer_weight/df_ans.csv')

print('————预测结果分析————')
for i in range(len(ans)):
    print('预测错误图片路径为：{}        真实标签为：{}        预测结果为：{}'.format(ans[i][0], ans[i][1], ans[i][2]))

