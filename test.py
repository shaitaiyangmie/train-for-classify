import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import torch.nn as nn
import os
# from train import list_train_name
import pandas as pd

print('————正在运行test.py文件————')
print('————加载数据中————')
root_test = 'D:/breast cancer/'
list_test_name = os.listdir(root_test)

list_test_img = []
for name in list_test_name:
    img = Image.open(os.path.join(root_test, name))
    list_test_img.append(np.array(img))


print('img个数为{}'.format(len(list_test_img)))
print('————数据加载完毕————')

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class TestData(Dataset):
    def __init__(self, x, transform):
        self.X = x
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_idx = self.X[idx]
        x_idx = self.transform(Image.fromarray(x_idx).convert('RGB')).type(torch.FloatTensor)
        return x_idx


test_set = TestData(list_test_img, transform_test)
x = test_set[0]
print('输入数据的size为{}'.format(x.size()))

test_loader = DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

print('————数据实例化完毕————')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
model.fc = nn.Linear(512, 12)
model.load_state_dict(torch.load('D://新建文件夹/植物识别/weight.pth'))
model = model.to(DEVICE)
sample_sub = pd.read_csv('D://新建文件夹/植物识别/plant-seedlings-classification/sample_submission.csv')
print('————模型加载完毕————')

print('————开始预测————')
total = 0
correct = 0
list_pred = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(test_loader):
        data = data.to(DEVICE)
        y_pred = model(data)
        pred = y_pred.argmax(dim=1)
        # pred = y_pred.argmax(dim=1)也ok
        list_pred += pred.cpu().numpy().tolist()
# array_pred = np.array(list_pred)
# print('list_pred的shape是{}'.format(array_pred.shape))

label2cate = {}
root_train = 'D://新建文件夹/植物识别/plant-seedlings-classification/train'
list_train_cate = os.listdir(root_train)
for i in range(len(list_train_cate)):
    if i not in label2cate:
        label2cate[i] = list_train_cate[i]

sample_sub['species'] = list(map(lambda x: label2cate[x], list_pred))
print('————预测结束————')

sample_sub.to_csv('D://新建文件夹/植物识别/submission.csv')
print(sample_sub.head())
print('————预测结果保存完毕————')
