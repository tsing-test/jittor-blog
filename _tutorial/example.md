---
layout: wiki
title: "Example: Model definition and training"
categories: Tutorial
---

The following example shows how to model a two-layer neural network step by step and train from scratch In a few lines of Python code.

```python
import jittor as jt
import numpy as np
```

The following code defines our model, which is a two-layer neural network. The size of hidden layer is 10. and the activation function is relu. **var_scope** is used to reuse parameters and prevent duplicate names in the model.

```python
@jt.var_scope('model', unique=True)
def model(x):
    x = linear(x, 10)
    x = relu(x)
    x = linear(x, 1)
    return x
```

The following code defines the activation function and linear layer. The matrix multiplication operator which is required by the linear layer, is consists of three **meta-operators**, which are broadcast and reduce operators. There are many benefits when the matrix multiplication is defined by meta-operations. For example, it is intuitive and friendly to the back-end. The back-end is allowed to explore more optimization strategies compare to hand-tuned operators.

```python
@jt.var_scope('linear')
def linear(x, n):
    w = jt.make_var([x.shape[-1], n], init=lambda *a:
            (jt.random(*a)-0.5) / x.shape[-1]**0.5)
    b = jt.make_var([n], init=lambda *a: jt.random(*a)-0.5)
    return matmul(x, w) + b

def matmul(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n,m,k], dims=[2])
    b = b.broadcast([n,m,k], dims=[0])
    return (a*b).sum(dim=1)

def relu(x): return jt.maximum(x, 0.0)
```

At last, this model is trained from scratch. A simple gradient descent is used, and the loss function is L2 distance. The taining process is asynchronous for efficiency. jittor calculates the gradients and applies graph- and operator-level optimizations via **unify IR graph** and **jit analyzer**.
In this example, multiple optimizations can be used, including: **operator fusion**, the activation function and loss function can be fused into the first and second linear layers; Three meta-operators in matrix multiplication could also be fused. **Parallelism**, it can improve performance of compute-intensive operations on modern multi-core CPUs and GPUs. The operator fusion is a graph-level optimization, and parallelism can be achieved in both graph- and operator-level.


```python
np.random.seed(0)
jt.set_seed(3)
n = 1000
batch_size = 50
base_lr = 0.05
# we need to stop grad of global value to prevent memory leak
lr = jt.float32(base_lr).name("lr").stop_grad()

def get_data(n):
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield jt.float32(x), jt.float32(y)

for i,(x,y) in enumerate(get_data(n)):
    pred_y = model(x).name("pred_y")
    loss = ((pred_y - y)**2).name("loss")
    loss_mean = loss.mean()

    ps = jt.find_vars('model')
    gs = jt.grad(loss_mean, ps)
    for p,g in zip(ps, gs):
        p -= g * lr
    # The above four lines is a simple gradient descent 
    # for demonstration. You can use 
    #       jt.nn.SGD('model', loss_mean, lr)
    # for convenience

    print(f"step {i}, loss = {loss_mean.data.sum()}")

# result is 0.0009948202641680837
result = 0.0009948202641680837
assert abs(loss_mean.data - result) < 1e-6
```