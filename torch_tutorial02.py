"""
使用torch实现二分类的功能Step1：仅将numpy改为torch，并优化损失函数、添加重构
"""
import torch
import torch.nn.functional as F
from torch import nn


# 训练集和验证集，四组数据，三维向量
x_train, y_train, x_valid, y_valid = [[1.0,2,3], [2,3,4], [3,4,5], [4,5,6]], [0, 0, 1, 1], [[7.0,9,8], [0,0,0]], [1, 0]
# 改变格式
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)  )
# 重构
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones((3,2)))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, xb):
        return xb @ self.weights + self.bias

# 实例化模型
model = Mnist_Logistic()
# 损失函数，交叉熵
loss_func = F.cross_entropy
# 准确率
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
# 观察结果
# bs = 2  # batch size
# xb = x_train[0:bs]  # a mini-batch from x
# yb = y_train[0:bs]
# print(loss_func(model(xb), yb), accuracy(model(xb), yb))

# 多次迭代
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for
bs = 2  # batch size
for epoch in range(epochs):
    for i in range((x_train.shape[0] - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        print('epoch=', epoch, 'batchnum=', i, 'loss=', loss, '\nweights=\n', model.weights)

        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= p.grad * lr
            model.zero_grad()

print('loss=', loss_func(model(x_train), y_train), 'accuracy=', accuracy(model(x_valid), y_valid), '\nweights=\n', model.weights, '\nbias=', model.bias)
