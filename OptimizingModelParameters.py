'''
参考Pytorch官方文档：https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
这个文档开始正式训练模型
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 超参数 --- 超参数调整官方文档：https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
# 优化循环 --- 有了超参数，就要训练并优化模型，这个过程称为优化循环，包含了两个部分
# 1) 训练循环：迭代训练集尽量收敛到最优参数
# 2) 验证/测试循环：迭代测试集测试是否模型性能提升

# 迭代训练集时包含了计算误差和更新权重参数两个步骤
# 误差使用 Loss Function 来表示，pytorch的误差函数有：https://pytorch.org/docs/stable/nn.html#loss-functions
# 优化器Optimizer：用来更新权重参数减少误差，不同的优化器效果不同，pytorch官方提供的优化器在：https://pytorch.org/docs/stable/optim.html

# 关于优化器的说明
# 不同优化器之间其实是权重参数更新公式不同，这样的话loss function就会有不同的收敛速度和准确度
# 参见：https://zhuanlan.zhihu.com/p/55150256
# https://www.jiqizhixin.com/articles/2021-01-05-9

def train_loop(dataloader, model, loss_fn, optimizer):
    '''
    训练模型：循环我们的优化代码
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :return:
    '''
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # BackPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print("loss:{}  [{}/{}]".format(loss, current, size))

def test_loop(dataloader, model, loss_fn):
    '''
    使用测试数据测试刚刚训练出来的model的性能
    :param dataloader:
    :param model:
    :param loss_fn:
    :return:
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 实例化模型开始训练
model = NeuralNetwork()
learning_rate = 1e-3
batch_size = 64
epochs = 10
loss_fn = nn.CrossEntropyLoss() # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print("Epoch ",t+1, "\n")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done")