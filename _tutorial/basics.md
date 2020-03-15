---
layout: wiki
title: "Basics: Op, Var and var_scope"
categories: Tutorial
---

# Basics: Op, Var and var_scope
To train your model with jittor, there are only three main concept you need to know:
* Op: numpy like operator
* Var: output variable of Op
* var_scope: namescopes used for variable storage

## Op
First, let's get started with Op. jittor'op is simular with numpy. Let's try some operations. We create Var `a` and `b` via operation `jt.float32`, and add them. Printing those variables shows they have the same shape and dtype.

```python
import jittor as jt
a = jt.float32([1,2,3])
b = jt.float32([4,5,6])
c = a*b
print(a,b,c)
```

if you want to know all the operation which jittor supports. try `help(jt.ops)`. All the operation you found in `jt.ops.xxx`, can be used via alias `jt.xxx`

```python
help(jt.ops)
```

## Var
Var is the output variable of operator. Computation process in Jittor is asynchronous for optimization. If you want to access the data, `Var.data` can be used for synchronous data accessing.

```python
print(c.data)
```

Beside that, All the operators we used `jt.xxx(Var, ...)` have alias `Var.xxx(...)`. For example:

```python
c.max() # alias of jt.max(a)
c.add(a) # alias of jt.add(c, a)
c.min(keepdims=True) # alias of jt.min(c, keepdims=True)
```

And we can give the variable a name

```python
c.name('c')
print(c.name())
```

## var_scope
`var_scope` is a namescope used for variable storage. In the code below, we create some variable in var_scope.

```python
# jt.clean is used for clean all the scopes and variables
jt.clean()

shape = [10]
with jt.var_scope('a'):
    x = jt.make_var(shape, init=jt.random, name="x")
    
with jt.var_scope('a'):
    x = jt.make_var(shape, init=jt.random, name="x")
    
params = jt.find_vars()
names = [ p.name() for p in params ]
print(names)
```

In the code above, we create two scopes, named `a_0` and `a_1`, and two variables under those two scopes. The full names of those variables are concat with scope names. Those two variables are initialized with `jt.random`, and shape is `[10]`, dtype is `float` by default.

Let's change a little bit, add an argument `unique=True`

```python
jt.clean()

shape = [10]
with jt.var_scope('a', unique=True):
    x = jt.make_var(shape, init=jt.random, name="x")
    
with jt.var_scope('a', unique=True):
    x = jt.make_var(shape, init=jt.random, name="x")
    
params = jt.find_vars()
names = [ p.name() for p in params ]
print(names)
```

As you can see, we only create a single variable. This is because without generate index in the scopes name, the first and the second scope are the same scope. It makes the variables `x` are the same variable.

With this mechanism, we can create parameters in model definition.

```python
jt.clean()
f32 = jt.float32

@jt.var_scope('model', unique=True)
def model(x):
    x = linear(x, 10)
    x = relu(x)
    x = linear(x, 1)
    return x

@jt.var_scope('linear')
def linear(x, n):
    w = jt.make_var([x.shape[-1], n], name="w", init=lambda *a:
            (jt.random(*a)-f32(0.5)) / f32(x.shape[-1])**f32(0.5))
    b = jt.make_var([n], name="b", init=lambda *a: jt.random(*a)-f32(0.5))
    return matmul(x, w) + b

def matmul(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n,m,k], dims=[2])
    b = b.broadcast([n,m,k], dims=[0])
    return (a*b).sum(dim=1)

def relu(x): return jt.maximum(x, f32(0))

y = model(jt.random([10, 100]))

params = jt.find_vars()
names = [ p.name() for p in params ]
print(names)
```

In this model definition, we create two linear layers and four parameters in total. The parameters can be obtained by `jt.find_vars()`. And you can manipulate then in any way you want(e.g. calculate the grad via `jt.grad` and update them).