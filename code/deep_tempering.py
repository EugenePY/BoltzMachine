from rbm import RBM

import theano
import theano.tensor as T


class Tempering(RBM):
    "Deep Tempering for training the Deep Boltzmann Machine(UNDONE)"
    def __init__(self, input):
        '''
        input: dense matrix tensor variable.
        '''
        self.x = input

    def Neg_swap_deep(self, x, y):
        pass

    def Neg_swap_parrallel(self, x, y):
        pass
if __name__ == "__main__":
    pass
