"""
使用二层神经网络及numpy，解决一个二分类问题：sum(2x+1) > 25的为1，<=25 为 0
"""
import numpy as np


# 训练集和验证集，四组数据，三维向量
x_train, y_train, x_valid, y_valid = [[1.0,2,3], [2,3,4], [3,4,5], [4,5,6]], [0, 0, 1, 1], [[7.0,9,8], [0,0,0]], [1, 0]
# 改变格式
x_train, y_train, x_valid, y_valid = map(
    np.array, (x_train, y_train, x_valid, y_valid)  )
# 构造权重和偏置，全连接层，三维降到二维
weights = np.ones((3,2))
bias = np.zeros(2)
# 构造模型及激活函数softmax，加log原因是后面的交叉熵的计算用到了log，在这里能简化计算
def log_softmax(x):
    return x - np.log(np.exp(x).sum(-1)).reshape(-1,1)
    # return np.exp(x) / np.exp(x).sum(-1).reshape(-1,1)  # 不加log
def model(xb):
    return log_softmax(xb @ weights + bias)  # @表示点乘
# 观察结果
# bs = 2  # batch size
# xb = x_train[0:bs]  # a mini-batch from x
# preds = model(xb)  # predictions
# print('preds=', preds, preds.shape)
# 损失函数，交叉熵
def loss_func(input, target):
    return -input[range(target.shape[0]), target].mean()
    # return -np.log(input[range(target.shape[0]), target]).mean() # 激活函数不使用log
# 观察结果
# yb = y_train[0:bs]
# print('loss_func=', loss_func(preds, yb))
# 准确率
def accuracy(out, yb):
    preds = np.argmax(out, axis=1)
    return ((preds == yb)*1.0).mean()
# print('accuracy=', accuracy(preds, yb))
# 反向传播
def backpropagation(input_x, output_z, target):   
    flag_dz = np.zeros_like(output_z)
    for i in range(len(target)): flag_dz[i][target[i]]=1 
    dl_dy = np.exp(output_z) - flag_dz
    dy_dw = input_x
    weights_grad = np.zeros_like(weights)
    bias_grad = np.zeros_like(bias)
    for i in range(dl_dy.shape[0]):
        weights_grad += (dl_dy[i].reshape(-1,1) @ dy_dw[i].reshape(1,-1)).T
        bias_grad += dl_dy[i] * 1.0
    weights_grad = weights_grad / dy_dw.shape[0]
    bias_grad = bias_grad / dy_dw.shape[0]
    return weights_grad, bias_grad
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

        weights_grad = np.zeros_like(weights)
        bias_grad = np.zeros_like(bias)        
        weights_grad, bias_grad = backpropagation(xb, pred, yb)
        print('epoch=', epoch, 'batchnum=', i, 'loss=', loss, '\nweights=\n', weights)
    
        weights -= weights_grad * lr
        bias -= bias_grad * lr

print('loss=', loss_func(x_train, y_train), 'accuracy=', accuracy(x_valid, y_valid), '\nweights=\n', weights, '\nbias=', bias)

