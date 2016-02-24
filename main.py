'''
Prototype implementation of a Neural Tensor Network Layer-Socher et al.'13
'''
import numpy as np
import theano

from theano import tensor

floatX = theano.config.floatX


class NTN_layer(object):
    def __init__(self, nin1, nin2, nout,
                 lr_dim=None, **kwargs):
        self.nin1 = nin1  # dimension of 1st vector
        self.nin2 = nin2  # dimension of 2nd vector
        self.nout = nout  # dimension of output (tensor slices)
        self.lr_dim = lr_dim  # low rank dimension
        self._init_params(**kwargs)

    def _init_params(self, scale=0.01, name='ntn'):
        n = self.nin1 + self.nin2
        nout = self.nout
        lr_dim = self.lr_dim
        if lr_dim is None:
            self.params = {
                't': theano.shared(
                    scale * np.random.randn(self.nout, n, n).astype(floatX))
                }
        else:
            self.params = {
                'tleft': theano.shared(
                    scale * np.random.randn(nout, n, lr_dim).astype(floatX)),
                'tright': theano.shared(
                    scale * np.random.randn(nout, lr_dim, n).astype(floatX)),
                'tdiag': theano.shared(
                    scale * np.random.randn(nout, n).astype(floatX))
                }

    def fprop(self, x1, x2):
        """
        x1 : (batch_size, nin1)
        x2 : (batch_size, nin2)
        """
        x = tensor.concatenate([x1, x2], axis=1)
        if self.lr_dim is None:
            return self._fprop_full(x)
        return self._fprop_lr(x)

    def __tensor_bilin(self, x_, t_):
        # a bit tricky - right product is done manually by:
        #   1. hadamard product with batch
        #   2. sum along the concatenated dimension
        return (x_.dot(t_) * x_[:, None, :]).sum(2)

    def _fprop_full(self, x):
        t = self.params['t']
        return self.__tensor_bilin(x, t)

    def _fprop_lr(self, x):
        # probably i'm overcomplicating low rank implementation :/
        tleft = self.params['tleft']
        tright = self.params['tright']
        tdiag = self.params['tdiag']

        # low rank part
        lr_tensor = theano.tensor.stack(
            [tleft[i].dot(tright[i]) for i in range(self.nout)],
            axis=0)
        lr_prod = self.__tensor_bilin(x, lr_tensor)

        # i think this is needed just to ensure low rank multiplication
        # result to be non-singular, adding some data dependent bias.
        diag_tensor = theano.tensor.stack(
            [tensor.nlinalg.alloc_diag(tdiag[i, :]) for i in range(self.nout)],
            axis=0)
        lr_bias = self.__tensor_bilin(x, diag_tensor)

        return lr_prod + lr_bias


if __name__ == "__main__":

    batch_size = 5
    vec1_dim = 6
    vec2_dim = 4
    tensor_dim = 11
    lr_dim = 2

    x1 = theano.tensor.matrix('x1', dtype=floatX)
    x2 = theano.tensor.matrix('x2', dtype=floatX)
    x1_ = np.random.randn(batch_size, vec1_dim).astype(floatX)
    x2_ = np.random.randn(batch_size, vec2_dim).astype(floatX)

    print 'Initializing full-rank NTN layer'
    ntn = NTN_layer(vec1_dim, vec2_dim, tensor_dim)
    z = ntn.fprop(x1, x2)

    print 'Compiling theano function for full rank NTN'
    f = theano.function([x1, x2], z)

    print 'Full rank NTN:'
    print f(x1_, x2_).shape

    print 'Initializing low-rank NTN layer'
    ntn2 = NTN_layer(vec1_dim, vec2_dim, tensor_dim, lr_dim=lr_dim)
    z2 = ntn2.fprop(x1, x2)

    print 'Compiling theano function for low rank NTN'
    f2 = theano.function([x1, x2], z2)

    print 'Low rank NTN:'
    print f2(x1_, x2_).shape
