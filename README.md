作者：唯物链丶
链接：https://zhuanlan.zhihu.com/p/102127047
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

XLA: XLA的做法实际上是将从deep learning framework训练得到的computational graph中的每个node都抽象成为最基本的线性代数操作，然后调用目前经过专家手工优化的库比如说，cpu上的eign，gpu上的cudnn来提交表现性能。比如，对于dot product这个op，xla则采用了向量化的LLVM IR并且上层仅仅针对TensorFlow下层针对TPU，开源的代码不多，仅仅知道个大概即可。
TVM: TVM则是通过将computational graph中的node结合loopy进行优化（因为在deep learning中，大部分需要我们优化的工作都是多重循环）lower到halide IR，然后通过Halide IR来做cuda，opencl，metal等不同backend的支持。
Tensor Comprehensions: TC其实是为神经网络提供了一个新的abstraction，以至于让JIT这样的compiler可以通过一定的算法来找到最优的执行plan，然后这个plan又被根据你指定的不同backend来generate成不同的code，其实TC的好处很明显，就是能够帮我们找到一些现在不存在的op，并且通过将其高效的实现出来。
Glow: Glow的思路很简单，和上述这些deep learning compiler一样， 有一个或多个IR，然后在低阶IR中，glow会将复杂的op通过一系列简单的线性代数源语来实现。

关于Glow的motivation其实也是很简单的，也就是说，在得到一张computational graph后，我们仅仅通过一层编译手段，将graph中的每个op都变成由一系列loop和其他低阶IR这样的优化显然是不够的。我们还必须有考虑到高阶的IR。比如对于一个多重for-loop语句来看，我们不能通过一个传统的编译器来帮我们解决这个问题，多层for-loop的优化，他们是做不到的。此时，针对这个多重for-loop（卷积）我们就可以定义一种高阶的IR，例如将data的format定义为tensor（N, C, H, W）的格式，从而帮我们完成相应的optimization。有了这个motivation，glow就被设计出来了，只要让compiler的前几个stage是target-independent的，让他更加倾向于我们所需要解决的任务的data type就行。但是当compiler越接近底层的不同hardware platforms的时候，我们的低阶IR就要更加specific到硬件架构的设计了。# Dive into Deep Learning Compiler

- Website (with CDN): https://tvm.d2l.ai/
- Website without CDN (any change will display immediately): http://tvm.d2l.ai.s3-website-us-west-2.amazonaws.com/ 

## How to contribute

- Roadmap https://docs.google.com/document/d/14Bgo9TgczROlqcTinS5-Y4hyig-ae0Rm8uIFQl9OAEA/edit?usp=sharing 
- Use Jupyter to edit the markdown files: http://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html#markdown-files-in-jupyter
- How to send a PR on Github: http://d2l.ai/chapter_appendix-tools-for-deep-learning/contributing.html
- Style guideline: https://github.com/d2l-ai/d2l-en/blob/master/STYLE_GUIDE.md



# TVM 基础
1. Data Types

        主要内容：介绍TVM创建Ops中Data Types相关内容。 
        TVM中Data Types（即dtype）的类型包括 'float16', 'float64', 'int8','int16', 'int64'。
        TVM Tensor类型转换使用方法 tensor.astype('int32')
        TVM Tensor转换为NDArray对象，tensor.asnumpy() 

2. Shapes

 
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
