---
layout: page
title: Download
description: 人越学越觉得自己无知
keywords: 下载, download
comments: false
menu: 下载
permalink: /download/
---

## Install

Jittor is written in Python and C++. It requires a compiler for JIT compilation, Currently, we support four compilers:

* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0) recommend
* GPU compiler (optional)
    * nvcc (>=10.0)

We provide single line command for quick installation the latest version of Jittor(Ubuntu>=16.04):

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
After execution, the script will show some environment variables you need to export.

If you use Jittor for CPU computing, we strongly recommend clang(>=8.0) as the back-end compiler of Jittor. Because some customized optimizations will be enabled.

We will show how to install Jittor in Ubuntu 16.04 step by step, Other Linux distributions may have similar commands.

### Step 1: Choose your back-end compiler

```bash
# g++
sudo apt install g++ build-essential libomp-dev

# OR clang-8
wget -O - https://apt.llvm.org/llvm.sh > /tmp/llvm.sh
bash /tmp/llvm.sh 8
```
if you choose icc as the back-end compiler, please go to the [offical icc install page](https://software.intel.com/en-us/cpp-compiler-developer-guide-and-reference-compiler-setup).

### Step 2: Install Python and python-dev
Jittor need python version >= 3.7.

```bash
sudo apt install python3.7 python3.7-dev
```

### Step 3: Run Jittor

The whole framework is compiled Just-in-time. Let's install jittor via pip

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
if the test is passed, your Jittor is ready.

### Optional Step 4: Enable CUDA
Using CUDA in Jittor is very simple, Just setup environment value `nvcc_path`

```bash
# replace this var with your nvcc location 
export nvcc_path="/usr/local/cuda/bin/nvcc" 
# run a simple cuda test
python3.7 -m jittor.test.test_cuda 
```
if the test is passed, your can use Jittor with CUDA by setting `use_cuda` flag.

```python
import jittor as jt
jt.flags.use_cuda = 1
```

### Optional Step 5: Run full tests
To check the integrity of Jittor, we can run full tests.

```bash
python3.7 -m jittor.test -v
```
if those tests are failed, please report bugs for us, and feel free to contribute ^_^
