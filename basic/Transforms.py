'''
参见pytorch官方文档：https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
Transforms官方文档：https://pytorch.org/vision/stable/transforms.html
'''

# 以FashionMNIST为例
# 我们需要把原来PIL Image format的样本特征转化为标准化（或者说归一化）的tensors
# 同样，我们需要把labels转化为 one-hot encoded tensors. --- 关于 one-hot encoding 参见：https://www.educative.io/blog/one-hot-encoding
# 上述特征和labels两步转换，我们主要使用 ToTensor 和 Lambda
import torch
from torchvision import datasets
from  torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(), # ToTensor()将PIL image转化为FloatTensor，并且像素点范围在[0.,1.]间
    target_transform=Lambda(lambda y: torch.zeros(10,dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    # user-defined lambda函数：首先定义一个size为10的zero tensor，然后使用tensor的scatter方法设置index处为value为1 --- 这一步是在进行 one-hot encoding
)
# 【注】ToTensor() 和 Lambda 都是参考的 Tramsforms 官方文档

# 总结
# 1. 在使用机器学习算法训练模型时，大多数情况下数据并不会符合算法需要的数据形式，所以Transforms的作用就是将数据转化成适合训练的数据形式。
# 2. 要想转换数据（使用Transforms）其实只需要使用pytorch框架中已经封装好的一些函数
# 3. Transforms 封装好的函数使用时是在 datasets类加载数据时的参数中使用，所有的torchvision的datasets都有transform和target_transform两个参数，其中前者负责对features进行数据格式转换，后者负责对labels进行数据形式转换。
# 4. 依然，万物可见官方文档，Transforms可以开箱即用的函数官方文档：https://pytorch.org/vision/stable/transforms.html

'''''''!!! transform还需要再看，官方文档中有两个转换的例子，可以试着跑一跑'''