from rbm import RBM
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np


class HiddenLayer(RBM):
    def __init__(self, input, n_in, n_out, rng=None,
                 W=None, vbias=None, hbias=None):
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        RBM.__init__(self, self.input, n_visible=self.n_in,
                     n_hidden=self.n_out, W=W,
                     vbias=vbias, hbias=hbias
                     )
        if rng is None:
            numpy_rng = np.random.RandomState(1234)
            rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        self.theano_rng = rng

        self.output = self.sample_h_given_v(self.input)[2]

        self.feedbackward = self.gibbs_vhv(self.input)[2]


if __name__ == '__main__':
    pass
