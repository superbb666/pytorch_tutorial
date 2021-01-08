"""
使用torch实现二分类的功能Step3：美化程序（函数封装一下）
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# 训练集和验证集，四组数据，三维向量
x_train, y_train, x_valid, y_valid = [[1.0,2,3], [2,3,4], [3,4,5], [4,5,6]], [0, 0, 1, 1], [[7.0,9,8], [0,0,0]], [1, 0]
# 改变格式
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)  )
# 参数部分
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for
bs = 2  # batch size
# 数据集封装
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
# 获取数据加载器
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
# 重构
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)  # 三维降两维，weights和bias初始值随机

    def forward(self, xb):
        return self.lin(xb)

# 实例化模型
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)
# 损失函数
loss_func = F.cross_entropy
# 计算loss及反向传播
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
# 多次迭代
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()  # 下面将进行train模式
        for xb,yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()  # 下面将进行eval模式
        with torch.no_grad():
            losses, nums = zip(
                    *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)
# 准确率
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# 使用模型
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
print('accuracy=', accuracy(model(x_valid), y_valid))