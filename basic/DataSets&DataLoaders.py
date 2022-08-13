'''
参考pytorch入门官方教程：https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
'''

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Load the Fashion-MNIST dataset from TorchVision
# Fashion-MNIST dataset 官方文档：https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 数据集迭代和可视化
# 像一个字典一样索引数据集
labels_map = {
    0:"T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# 使用matplotlib可视化一些样本
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# 自定义数据集类  --- 自定义数据集类必须包含三个函数：__init__, __len__, __getitem__
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_files, img_dir, transform=None, target_transform=None):
        '''
        实例化数据集对象时执行一次，这里初始化的 directory 包含图像(images)/标签文件(annotations file)/transforms（下一节再看是什么东西）
        :param annotations_files:
        :param img_dir:
        :param transform:
        :param target_transform:
        '''
        # pd.read_csv方法读出来的是一个DataFrame
        self.img_labels = pd.read_csv(annotations_files) # the labels of datasets are stored in CSV file annotations_file
        self.img_dir = img_dir # FashionMNIST images are stored in directory img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        '''
        返回数据集中 sample 的数量
        :return:
        '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''
        依据给出的索引idx从数据集中加载并返回一个sample
        基于index，首先会定位一个图像在磁盘上的位置，然后再使用read_image函数将其转换成tensor
        之后再从img_labels的csv文件中检索出来正确的label
        :param idx:
        :return:
        '''
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# 使用DataLoader准备训练使用的数据
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=50, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=50, shuffle=True)
# 【注】关于DataLoader的三点说明
# 1. Dataset与DataLoader的区别：Dateset每次只能加载数据集中的一个sample，但是使用DataLoader每次可以加载一个minibatch的数据（使用batch_size设置）
# 2. 参数shuffle是用来对每一个epoch进行“洗牌”的，也就是说每一个epoch中都会有很多的minibatch，经过shuffle后的每一epoch中的minibatch中的样本都会不同，这样可以防止过拟合
# 3. DataLoader内部使用了Python的multiprocessing的，这样加快了数据的检索

# 使用加载进DataLoader的数据进行迭代（训练模型）
# Display image and label
train_features, train_labels = next(iter(train_dataloader)) # next()函数：返回迭代器中的下一个项目
print("Feature batch shape:", train_features.size())
print("Labels batch shape:", train_labels.size())
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print("Label:", label)
# 【注】关于 iteration/epoch/minibatch 的体会
# 一个minibatch中包含了每次iteration的sample，也就是说，每次训练完一个minibatch称为一次iteration
# 一个epoch中包含了很多次iteration
# 都以sample为单位进行衡量即为：一个epoch包含所有的sample，一个iteration包含一个minibatch中所有的sample

# 总结：
# 1. torchvision 是一种基于pytorch的用于图像处理的library
# 2. Pytorch 作为一种框架，针对一些很厉害的官方开源数据集提供了访问类（dataloader 和 dateset），实现了针对不同数据集的函数，我们可以针对数据集使用不同的方法直接调用数据集
# 3. 万物皆可官方文档！！！！torchvision 库针对的数据集的官方文档：https://pytorch.org/vision/stable/datasets.html
# 4. Dataset子类和DataLoader类的不同：
# 1) Pytorch实现的所有开源数据库的接口都是Dataset的子类，也就是说，我们从外界加载数据库时时候Dataset类
# 2) DataLoader可以使用torch.multiprocessing方法并行加载多个samples，所以Dataset中加载的数据可以进一步加载到DataLoader中，也就是说要训练模型前的数据准备工作用DataLoader类
# DataLoader官方文档：https://pytorch.org/docs/stable/data.html