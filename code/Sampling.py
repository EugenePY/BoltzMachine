import theano
import theano.tensor as T
import numpy as np


class CorrelationSamping:
    "Sampling from a Corrlation Matrix using Gibbs Sampling(UNDONE)"
    def __init__(self, sigma):
        self.sigma = sigma
        self.p00 = 0.05

    def kernel(self, particle):
        pass

if __name__ == '__main__':
    pass
