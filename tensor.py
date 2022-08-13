'''
参考pytorch入门官方教程：https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
'''
import torch
import numpy as np

# 创建 tensor --- 都是使用的torch的不同的方法
# 方法一：直接从数据创建
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print("x_data=", x_data)
# 方法二：从numpy数组创建tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("x_np=", x_np)
# 方法三：使用一个已存在的tensor进行创建新tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print("x_ones=", x_ones)
# 【注】新的tensor只保留旧tensor的properties(shape,datatype)，不保留旧tensor的值
# 但是可以进行 overridden
x_rand = torch.rand_like(x_ones, dtype=torch.float32)
print("x_rand=", x_rand)
# 方法四：使用tuple规定一个tensor的dimensions，然后使用torch对应的方法创建
shape = (2,3,)
rand_tensor = torch.rand(shape) # .rand()方法创建随机数
ones_tensor = torch.ones(shape) # .ones()方法创建都为1
zeros_tensor = torch.zeros(shape) # .zeros()方法创建都为0
print("rand_tensor=",rand_tensor)
print("ones_tensor=",ones_tensor)
print("zeros_tensor=",zeros_tensor)


# tensor 的属性 --- 包括了tensor的shape, datatype,以及tensor存储在什么设备上
tensor = torch.rand(3,4)
print("shape of tensor:",tensor.shape)
print("DataType of tensor:",tensor.dtype)
print("Device tensor is stored on:",tensor.device)


# tensor的更多操作 --- 也就是torch的更多方法，包括了换位，索引，切片，数学运算，线性代数，随机抽样等等等一百多种
# 见：https://pytorch.org/docs/stable/torch.html#
# 【注】上方创建tensor以及查看tensor属性等也都是使用的torch的方法
# For example
# 将cpu上的tensor迁移到GPU上
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print("Device tensor is stored on:",tensor.device)
else:
    print("No CUDA")
# 切片
tensor1 = torch.ones(4,4)
print("First row:",tensor1[0])
print("First column:",tensor1[:,0])
print("Last column:",tensor1[:,-1])
tensor1[:,1] = 0
print("tensor1=",tensor1)
# 沿给定的维度连接tensor
tensor2 = torch.rand(2,3)
tensor3 = torch.rand(2,3)
t1 = torch.cat([tensor2,tensor3], dim=0)
print("Concatenated with dim0:",t1)
t2 = torch.cat([tensor2,tensor3], dim=1)
print("Concatenated with dim1:",t2)
# 算术运算 --- 这里只演示乘法
# torch.matmul()没有强制规定维度和大小，可以用利用广播机制进行不同维度的相乘操作
y1 = torch.matmul(torch.randn(10,3,4),torch.randn(4,5))
print("y1.shape=",y1.shape)
# 对应位相乘
y2 = torch.mul(torch.rand(3,2),torch.rand(3,2))
print("y2.shape=",y2.shape)
# 矩阵行列乘法
y3 = torch.mm(torch.rand(1,3),torch.rand(3,2))
print("y3.shape=",y3.shape)
# 将只含一个元素的tensor转化为python数值
agg = torch.randn(1)
print(agg, type(agg.item()))
tensor4 = torch.tensor([[1]])
print(type(tensor4.item()))
tensor5 = torch.tensor([[[1]]])
print(type(tensor5.item()))
# tensor 和 numpy之间的转换
# tensor to numpy
t = torch.ones(5)
print("t=", t)
n = t.numpy()
print("n=", n)

# 总结
# 1. tensor是一种类似于arrays和matrix的一种特殊的数据结构，和numpy中的 ndarray 很像很像
# 2. 在pytorch中，统一使用 tensor 作为模型的输入输出以及模型参数
# 3. 看到 tensor 相关方法，不会就看官方文档
# 牢记！！！万物皆可官方文档：https://pytorch.org/docs/stable/torch.html