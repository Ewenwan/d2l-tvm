# Dive into Deep Learning Compiler

- Website (with CDN): https://tvm.d2l.ai/
- Website without CDN (any change will display immediately): http://tvm.d2l.ai.s3-website-us-west-2.amazonaws.com/ 

## How to contribute

- Roadmap https://docs.google.com/document/d/14Bgo9TgczROlqcTinS5-Y4hyig-ae0Rm8uIFQl9OAEA/edit?usp=sharing 
- Use Jupyter to edit the markdown files: http://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html#markdown-files-in-jupyter
- How to send a PR on Github: http://d2l.ai/chapter_appendix-tools-for-deep-learning/contributing.html
- Style guideline: https://github.com/d2l-ai/d2l-en/blob/master/STYLE_GUIDE.md




1. Data Types
链接
主要内容：介绍TVM创建Ops中Data Types相关内容。
TVM中Data Types（即dtype）的类型包括 'float16', 'float64', 'int8','int16', 'int64'。
TVM Tensor类型转换使用方法 tensor.astype('int32')
TVM Tensor转换为NDArray对象，tensor.asnumpy()
2. Shapes
链接
主要内容：介绍TVM Ops中Shape相关内容，主要就是介绍可变shape的Variable如何创建。
定义变长向量
通过 n = tvm.var(name='n') 定义shape。
定义变量是通过 A = tvm.placeholder((n,), name='a') 实现。 
A = tvm.placeholder((n,), name='a')
B = tvm.placeholder((n,), name='b')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='c')
s = tvm.create_schedule(C.op)
tvm.lower(s, [A, B, C], simple_mode=True)
定义变长多维张量
需要首先指定 ndim，必须是常数。
定义张量shape时，通过 tvm.var 的数组来实现。
def tvm_vector_add(ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)])
    B = tvm.placeholder(A.shape)
    C = tvm.compute(A.shape, lambda *i: A[i] + B[i])
    s = tvm.create_schedule(C.op)
    return tvm.build(s, [A, B, C])
tvm.var 默认是创建 int32 类型的标量scalar。
3. Index and Shape Expressions
链接
主要内容：介绍几个Index以及Shape相关的操作。
矩阵转置
n = tvm.var('n')
m = tvm.var('m')
A = tvm.placeholder((n, m), name='a')
B = tvm.compute((m, n), lambda i, j: A[j, i], 'b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
2D array 转为 1D array
B = tvm.compute((m*n, ), lambda i: A[i//m, i%m], name='b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
1D array 转换为任意大小的 2D array
p, q = tvm.var('p'), tvm.var('q')
B = tvm.compute((p, q), lambda i, j: A[(i*q+j)//m, (i*q+j)%m], name='b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
切片操作，即numpy中的 a[bi::si, bj::sj] 操作
bi, bj, si, sj = [tvm.var(name) for name in ['bi', 'bj', 'si', 'sj']]
B = tvm.compute(((n-bi)//si, (m-bj)//sj), lambda i, j: A[i*si+bi, j*sj+bj], name='b')
s = tvm.create_schedule(B.op)
mod = tvm.build(s, [A, B, bi, si, bj, sj])
4. Reduction Operations
链接
主要内容：实现归约操作。
归约操作就是沿着输入tensor的某个axis进行某个操作，如求和、平均等。
实例一：将二维矩阵通过累加转换为一维向量
n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m), name='a')
j = tvm.reduce_axis((0, m), name='j')
B = tvm.compute((n,), lambda i: tvm.sum(A[i, j], axis=j), name='b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
实例二：Commutative Reduction（可交换的归约操作）
可以定义一些没有内置的Reduction操作。
comp = lambda a, b: a * b
init = lambda dtype: tvm.const(1, dtype=dtype)
product = tvm.comm_reducer(comp, init)
n = tvm.var('n')
m = tvm.var('m')
A = tvm.placeholder((n, m), name='a')
k = tvm.reduce_axis((0, m), name='k')
B = tvm.compute((n,), lambda i: product(A[i, k], axis=k), name='b')
s = tvm.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
5. Conditional Expression: if-then-else
链接
目标：实现条件表达式，类似 np.where 的功能。
感想：没啥好说的，就是介绍了一个API。
n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m))
B = tvm.compute(A.shape, lambda i, j: tvm.if_then_else(i >= j, A[i,j], 0.0))
b = tvm.nd.array(np.empty_like(a))
s = tvm.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))
mod = tvm.build(s, [A, B])
mod(tvm.nd.array(a), b)
6. Truth Value Testing: all and any
链接
目标：实现类似 np.any 和 np.all 的功能。
感想：就是俩API
import numpy as np
import d2ltvm
import tvm

p = 1 # padding size
n, m = tvm.var('n'), tvm.var('m')
A = tvm.placeholder((n, m), name='a')
B = tvm.compute((n+p*2, m+p*2),
                lambda i, j: tvm.if_then_else(
                    tvm.any(i<p, i>=n+p, j<p, j>=m+p), 0, A[i-p, j-p]),
                name='b')
7. 相关TVM API
tvm.var(name='n')
def var(name="tindex", dtype=int32)
作用：定义标量向量。
tvm.reduce_axis((0, m), name='j')
def reduce_axis(dom, name="rv")
作用：定义numpy类似 [s:e] 的功能。
tvm.comm_reducer(comp, init)
def reduce_axis(dom, name="rv")
作用：用于自定义reduction操作。 
tvm.compute(A.shape, lambda i, j: tvm.if_then_else(i >= j, A[i,j], 0.0))
def if_then_else(cond, t, f)
作用：实现类似 np.where 的功能。
tvm.any(i<p, i>=n+p, j<p, j>=m+p)
def any(*args)
作用：实现类似any的功能。
