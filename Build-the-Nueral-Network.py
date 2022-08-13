'''
本篇参考官方文档：https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
构建自己的神经网络用到的所有方法都在此官方文档中：https://pytorch.org/docs/stable/nn.html
'''

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 选择一个device进行训练
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# 定义神经网络子类 --- 神经网络本身是一个module，它包含很多层，每一个层又是一个module，在pytorch中，我们定义每一个module为nn.Module的子类。这样循环定义
class NeuralNetwork(nn.Module):
    def __init__(self):
        '''
        初始化神经网络层
        '''
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # Sequential官方文档：https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html，可以用来传递层之间的值
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

model = NeuralNetwork().to(device)
print(model)
# 给模型传入数据
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print("Predicted class:", y_pred)

# 分解上述神经网络，看各层作用
input_image = torch.rand(3,28,28) # 3代表的是一个minibatch中有3个samples
print(input_image.size())
# nn.Flatten --- 将2D 28*28的image转化为包含784个像素点的array
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
# nn.Linear --- 用其内部的weights和biases对输入数据进行线性转化
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
# nn.ReLU --- 非线性激活函数，线性变换后使用非线性变换，帮助神经网络学习各种各样现象
print("Before ReLU:", hidden1)
hidden1 = nn.ReLU()(hidden1)
print("After ReLU:", hidden1)
# nn.Sequential --- 是一个装着modules的顺序容器，数据按顺序通过它内部的所有modules，使用它可以快速创建一个网络（前边说的，网络的每一层都是一个module）
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
# 【注】Sequential中的参数放的都是module，也就是操作
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print("logits=", logits)
# nn.Softmax --- Softmax函数对n维tensor进行缩放，使其每个元素在[0,1]范围内，并且所有元素的和为1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)

# 模型参数 --- 上边说进行线性计算是使用模型内部的weights和biases，这里使用两个方法来查看模型参数
print("Model structure:",model)
for name, param in model.named_parameters():
    print("Layer:{}, Size:{}, Values:{}".format(name, param.size(), param[:2]))

# 总结
# Pytorch中神经网络本身是一个module，神经网络包含的很多层中每一层也都是一个module
# Pytorch中对于这些module都当成是 nn.Module 的子类，我们基于nn.Module搭建神经网络
# Again! 万物皆可官方文档，nn.Module的官方文档：https://pytorch.org/docs/stable/nn.html