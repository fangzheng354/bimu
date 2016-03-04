# http://deeplearning.net/software/theano/tutorial/using_gpu.html

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
#printf.maker.fgraph.toposort()
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
with open("gputest", "w") as outf:
    outf.write("Looping {} times took {} secs\n".format(iters, t1 - t0))
    outf.write("{}\n".format(r))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        outf.write('Used the cpu')
    else:
        outf.write('Used the gpu')
