"""
使用torch实现二分类的功能Step2：增加优化器、数据集切片（数据集重构、数据加载器重构）、添加验证
"""
import torch
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
# 数据集封装及切片：将x和y放到一起、按batch切开
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)  #shuffle：使得每个epoch的batch都不同
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

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

model, opt = get_model()
# 损失函数，交叉熵
loss_func = F.cross_entropy
# 准确率
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# 多次迭代
for epoch in range(epochs):
    model.train()  # 下面将进行train模式
    for xb,yb in train_dl:  #替代： for i in range((x_train.shape[0] - 1) // bs + 1): xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        print('epoch=', epoch, 'loss=', loss)

        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()  # 下面将进行eval模式
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
    print(epoch, valid_loss / len(valid_dl))

print('loss=', loss_func(model(x_train), y_train), 'accuracy=', accuracy(model(x_valid), y_valid))

