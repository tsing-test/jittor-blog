---
layout: home
css: ['pages/index.css']
comments: false
---

# Jittor: a Just-in-time(JIT) deep learning framework


Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model.


The front-end language is Python. Module Design is used in the front-end, like PyTorch and Keras. The back-end is implemented py high performance language, such as CUDA,C++.

The following example shows how to model a two-layer neural network step by step and train from scratch In a few lines of Python code.

```python
import jittor as jt
from jittor import Module
from jittor import nn
class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.Relu() 
        self.layer2 = nn.Linear(10, 1)
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def get_data(n): # generate random data for training test.
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield jt.float32(x), jt.float32(y)

model = Model()
learning_rate = 0.1
optim = nn.SGD(model.parameters(), learning_rate)

for i,(x,y) in enumerate(get_data(n)):
    pred_y = model(x)
    loss = ((pred_y - y)**2)
    loss_mean = loss.mean()
    optim.step (loss_mean)
    print(f"step {i}, loss = {loss_mean.data.sum()}")
```

## Contents

* [Quickstart](#quickstart)
* [Install](#install)
* [Tutorial](#tutorial)
* [Contributing](#contributing)
* [The Team](#theteam)
* [License](#license)

## Quickstart

We provide some jupyter notebooks to help you quick start with Jittor.

* [Example: Model definition and training][1]
* [Basics: Op, Var][2]
* [Meta-operator: Implement your own convolution with Meta-operator][3]

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

## Tutorial

In the tutorial section, we will briefly explain the basic concept of Jittor.

To train your model with Jittor, there are only three main concepts you need to know:
* Var: basic data type of jittor
* Operations: Jittor'op is simular with numpy

### Var

Var is the basic data type of jittor. Computation process in Jittor is asynchronous for optimization. If you want to access the data, `Var.data` can be used for synchronous data accessing.

```python
import jittor as jt
a = jt.float32([1,2,3])
print (a)
print (a.data)
# Output: float32[3,]
# Output: [ 1. 2. 3.]
```

And we can give the variable a name

```python
c.name('c')
print(c.name())
# Output: c
```


### Operations

First, let's get started with Op. Jittor'op is simular with numpy. Let's try some operations. We create Var `a` and `b` via operation `jt.float32`, and add them. Printing those variables shows they have the same shape and dtype.

```python
import jittor as jt
a = jt.float32([1,2,3])
b = jt.float32([4,5,6])
c = a*b
print(a,b,c)
print(type(a), type(b), type(c))
# Output: float32[3,] float32[3,] float32[3,]
# Output: <class 'jittor_core.Var'> <class 'jittor_core.Var'> <class 'jittor_core.Var'>
```
Beside that, All the operators we used jt.xxx(Var, ...) have alias Var.xxx(...). For example:

```python
c.max() # alias of jt.max(a)
c.add(a) # alias of jt.add(c, a)
c.min(keepdims=True) # alias of jt.min(c, keepdims=True)
```

if you want to know all the operation which Jittor supports. try `help(jt.ops)`. All the operation you found in `jt.ops.xxx`, can be used via alias `jt.xxx`

```python
help(jt.ops)
# Output:
#   abs(x: core.Var) -> core.Var
#   add(x: core.Var, y: core.Var) -> core.Var
#   array(data: array) -> core.Var
#   binary(x: core.Var, y: core.Var, op: str) -> core.Var
#   ......
```

### More

If you want to know more about Jittor, please check out the notebooks below:

* Quickstart
    * [Example: Model definition and training][1]
    * [Basics: Op, Var][2]
    * [Meta-operator: Implement your own convolution with Meta-operator][3]
* Advanced
    * [Custom Op: write your operator with C++ and CUDA and JIT compile it][4]
    * Profiler: Profiling your model
    * Jtune: Tool for performance tuning


[1]: {{ site.url }}/tutorial/example/
[2]: {{ site.url }}/tutorial/basics/
[3]: {{ site.url }}/tutorial/meta_op/
[4]: {{ site.url }}/tutorial/custom_op/

Those notebooks can be started in your own computer by `python3.7 -m jittor.notebook`

## Contributing

Jittor is still young. It may contain bugs and issues. Please report them in our bug track system. Contributions are welcome. Besides, if you have any ideas about Jittor, please let us know.

## The Team

Jittor is currently maintained by Dun Liang, Guo-Ye Yang, Guo-Wei Yang, Wen-Yang Zhou and Meng-Hao Guo. If you are also interested in Jittor and want to improve it, Please join us!

## License

Jittor is Apache 2.0 licensed, as found in the LICENSE.txt file.


