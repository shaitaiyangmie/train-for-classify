import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image


root_train = 'D://新建文件夹/植物识别/plant-seedlings-classification/train'
# root_test = 'D://新建文件夹/植物识别/plant-seedlings-classification/test'
list_train_name = os.listdir(root_train)

name2label = {}
for name in list_train_name:
    if name not in name2label:
        name2label[name] = list_train_name.index(name)
print(name2label)

list_train_img = []
list_train_label = []
for cate in list_train_name:
    list_cur = os.listdir(os.path.join(root_train, cate))
    for name in list_cur:
        img = Image.open(os.path.join(root_train, cate, name))
        img = np.array(img)
        list_train_img.append(img)
        list_train_label.append(name2label[cate])
print('图片个数是{}个，label个数是{}'.format(len(list_train_img), len(list_train_label)))

transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
print('————数据放入列表完成————')
list_train_img = np.array(list_train_img)
list_train_label = np.array(list_train_label)
print('输入array的shape:', list_train_img.shape)


class Mydata(Dataset):
    # x,y都是list
    def __init__(self, x, y, transform):
        self.X = x
        self.Y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_idx = self.X[idx]
        y_idx = self.Y[idx]
        x_idx = self.transform(Image.fromarray(x_idx).convert('RGB'))
        y_idx = np.array(y_idx)
        y_idx = torch.from_numpy(y_idx).type(torch.LongTensor)
        return x_idx.type(torch.FloatTensor), y_idx


train_set = Mydata(list_train_img, list_train_label, transform)
x, y = train_set[0]
print('每个输入数据的size()', x.size())

train_loader = DataLoader(
    train_set,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
print('————数据实例化完成————')

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, len(list_train_name))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fun = nn.CrossEntropyLoss()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
model = model.to(DEVICE)
print('————模型初始化完成————')

print('————开始训练————')
for epoch in range(10):
    model = model.train()
    loss_sum = 0
    total = 0
    correct = 0
    for i, (data, label) in enumerate(train_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        y_pred = model(data)
        pred = y_pred.max(1, keepdim=True)[1]
        loss = loss_fun(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if i % 7 == 6:
            print('————Epoch:{}————Index:{}————Batch_Loss:{}————'.format(epoch, i, loss_sum / 7))
            loss_sum = 0
        total += label.shape[0]
        correct += pred.eq(label.view_as(pred)).sum().item()
    print('————准确率是{}————'.format(correct / total))
    model.eval()
print('————训练完成————')

torch.save(model.state_dict(), 'D://新建文件夹/植物识别/weight.pth')
print('————保存模型参数完成————')







