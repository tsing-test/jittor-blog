---
layout: wiki
title: "Jittor MNIST 分类教程"
categories: [图像分类, MNIST, 教程]
filename: 2020-03-15-mnistclassification
description: 使用 Jittor 对 MNIST 进行分类。
---

## MNIST介绍

MNIST 数据集可在 http://yann.lecun.com/exdb/mnist/ 下载, 它是一个对0到9十个数字进行分类的数据集。它包含了四个部分:

训练图像: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)

训练标签: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)

测试图像: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)

测试标签: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)

MNIST 数据集如下图所示


<img src="/images/tutorial/{{ page.filename }}/mnist.png">

## 使用 Jittor 对 MNIST 进行分类

> 1.首先第一步，需要引入相关的依赖，如下所示。 

```python
# classification mnist example 
import jittor as jt  # 将 jittor 引入
from jittor import nn, Module  # 引入相关的模块
import numpy as np
import sys, os
import random
import math 
from jittor import init
jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。
# 如果 jt.flags.use_cuda=1，表示使用GPU训练 如果 jt.flags.use_cuda = 0 表示使用 CPU
from jittor.dataset.mnist import MNIST 
#由于 MNIST 是一个常见的数据集，其数据载入已经被封装进 jittor 所以可以直接调用。
import matplotlib.pyplot as plt
import pylab as pl # 用于绘制 Loss 曲线 和 MNIST 数据

```

> 2.模型的定义：我们定义模型需要继承 jittor 的 Module 类。需要实现 `__init__` 函数和 `execute` 函数。`__init__` 用于定义模型由哪些操作组成， `execute` 函数定义了模型执行的顺序和模型的返回值。

```python
class Model (Module):
    def __init__ (self):
        super (Model, self).__init__()
        self.conv1 = nn.Conv (1, 32, 3, 1) # no padding
        
        self.conv2 = nn.Conv (32, 64, 3, 1)
        self.bn = nn.BatchNorm(64)

        self.max_pool = nn.Pool (2, 2)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear (64 * 12 * 12, 256)
        self.fc2 = nn.Linear (256, 10)
    def execute (self, x) : 
        # it's simliar to forward function in Pytorch 
        x = self.conv1 (x)
        x = self.relu (x)
        
        x = self.conv2 (x)
        x = self.bn (x)
        x = self.relu (x)
        
        x = self.max_pool (x)
        x = jt.reshape (x, [x.shape[0], -1])
        x = self.fc1 (x)
        x = self.relu(x)
        x = self.fc2 (x)
        return x

```

> 3.对模型进行训练。对模型训练需要定义训练时的超参数，以及需要定义训练过程。训练函数在 train 函数中定义，测试函数在 val 函数中定义。


```python
def train(model, train_loader, optimizer, epoch, losses, losses_idx):
    model.train()
    lens = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step (loss)
        losses.append(loss.data[0])
        losses_idx.append(epoch * lens + batch_idx)
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader) ,
                100. * batch_idx / len(train_loader), loss.data[0]))

def val(model, val_loader, epoch):
    model.eval()
    
    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}'.format(epoch, \
                batch_idx, len(val_loader),1. * batch_idx, acc))
    print ('Test Acc =', total_acc / total_num)
    
batch_size = 64
learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
epochs = 1
losses = []
losses_idx = []
train_loader = MNIST (train=True, batch_size=batch_size, shuffle=True)

val_loader = MNIST (train=False, batch_size=1, shuffle=False)

model = Model ()
optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
for epoch in range(epochs):
    train(model, train_loader, optimizer, epoch, losses, losses_idx)
    test(model, test_loader, epoch)


```

> 4.绘制 Loss 曲线 : 将 Loss 曲线进行可视化。

```python
pl.plot(losses_idx, losses)
pl.xlabel('Iterations')
pl.ylabel('Train_loss')

```

> 5.存储模型：模型训练完成需要存储起来,下面代码展示了 Jittor 如何进行存储模型。

```python
model_path = './mnist_model.pkl'
model.save(model_path)
```

> 6.加载模型并对模型进行测试，下面展示了 Jittor 如何加载模型，并对模型进行测试。

```python
def vis_img(img):
    np_img = img.data.reshape([28, 28])
    plt.imshow(np_img, cmap='gray') 
    

new_model = Model()
new_model.load_parameters(jt.load(model_path))
data_iter = iter(val_loader)
val_data, val_label = next(data_iter)
print (val_label.shape)
outputs = new_model(val_data)
prediction = np.argmax(outputs.data, axis=1)
print (prediction)
print (val_label)

vis_img(val_data)
```

