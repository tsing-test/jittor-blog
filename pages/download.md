---
layout: page
title: Download
description: 人越学越觉得自己无知
keywords: 下载, download
comments: false
menu: 下载
permalink: /download/
---

## 安装


Jittor使用Python和C ++编写。 它需要用于即时编译的编译器。当前，我们支持三种编译器：

* CPU 编译器 （需要下列至少一个）
    - g++ （>=5.4.0）
    - clang （>=8.0）推荐
* GPU 编译器（可选）
    - nvcc（>=10.0）


我们提供能快速安装最新版本Jittor的单行命令（Ubuntu> = 16.04）：

```bash
# install with clang and cuda
git clone https://git.net9.org/cjld/jittor.git && with_clang=1 with_cuda=1 bash ./jittor/script/install.sh
# install with clang
git clone https://git.net9.org/cjld/jittor.git && with_clang=1 bash ./jittor/script/install.sh
# install with g++ and cuda
git clone https://git.net9.org/cjld/jittor.git && with_gcc=1 with_cuda=1 bash ./jittor/script/install.sh
# install with g++
git clone https://git.net9.org/cjld/jittor.git && with_gcc=1 bash ./jittor/script/install.sh
```

执行后，脚本将显示一些需要导出的环境变量。


如果将Jittor用于CPU计算，则强烈建议使用clang（> = 8.0）作为Jittor的后端编译器。 因为Jittor会用到其中一些定制的优化。


我们将逐步演示如何在Ubuntu 16.04中安装Jittor，其他Linux发行版可能可以使用类似的命令。


### 步骤一：选择您的后端编译器

```bash
# g++
sudo apt install g++ build-essential libomp-dev

# OR clang-8
wget -O - https://apt.llvm.org/llvm.sh > /tmp/llvm.sh
bash /tmp/llvm.sh 8
```

### 步骤二：安装Python和python-dev


Jittor需要python的版本>=3.7。

```bash
sudo apt install python3.7 python3.7-dev
```


### 步骤三：运行Jittor


整个框架是及时编译的。 让我们通过pip安装jittor

```bash
git clone https://git.net9.org/cjld/jittor.git
sudo pip3.7 install ./jittor
export cc_path="clang-8"
# if other compiler is used, change cc_path
# export cc_path="g++"
# export cc_path="icc"

# run a simple test
python3.7 -m jittor.test.test_example
```

如果通过了测试，那么您的Jittor已经准备就绪。


### 可选步骤四：启用CUDA


在Jittor中使用CUDA非常简单，只需设置环境值`nvcc_path`

```bash
# replace this var with your nvcc location 
export nvcc_path="/usr/local/cuda/bin/nvcc" 
# run a simple cuda test
python3.7 -m jittor.test.test_cuda 
```

如果测试通过，则可以通过设置`use_cuda`标识符在Jittor中启用CUDA。

```python
import jittor as jt
jt.flags.use_cuda = 1
```


### 可选步骤五：进行完整测试


要检查Jittor的完整性，您可以运行完整的测试。

```bash
python3.7 -m jittor.test -v
```

如果这些测试失败，请为我们报告错误，我们十分欢迎您为Jittor做出贡献^ _ ^