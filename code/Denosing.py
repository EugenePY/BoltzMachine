# *-* coding: utf-8 *-*
# Author: 寇先元 R02323025 Dep. of Economics @ NTU

import numpy as np
from Metropolis_Hastings import MHasting

import cProfile


class Denosing(MHasting):
    '''
    This class has provide some function for Denosing a noisy image
    Provide the Simulated Annealing. And Finding the parameter of prior
    distribution. The Demo method provide some observation different update
    scheme will cause different results.
    '''

    def __init__(self, image):
        self.image = image.ravel()
        MHasting.__init__(self, self.y_given_x, n_vis=self.image.size)
        self.size = self.image.shape
        self.image_shape = image.shape
        self.p_filp = np.array([[0.625, 0.375],
                                [0.375, 0.625]])
        self.init = self.image

    def demo(self):
        '''
        compare the sequential update and the update all.
        '''
        pass

    def r(self, y, x, idx=None, T=0.01):
        return (self.l_x_given_y(y, idx) / self.l_x_given_y(x, idx)) * \
            self.Energy_diff(y, x, idx, T)

    def MH_accept(self, y, x, idx=None, T=None):
        r = self.r(y, x, idx, T)
        next_accept = (r > self.rng.uniform(size=self.size))
        accept = np.vstack([(1-next_accept).astype(bool), next_accept])
        return accept

    def sample_x_given_y(self, y):
        def state_to_index(state):
            '''
            State space is in {1, -1}
            '''
            return (state + 1)/2

        p = self.p_filp[1][list(state_to_index(y))]
        return np.random.binomial(1, p=p,
                                  size=self.size) * 2 - 1

    def proposal_sample(self, x=None, n_update=None):
        """
        Overided the method in MHasting
        """
        if n_update is None:
            n_update = self.size
        return self.rng.binomial(1, p=0.5, size=self.size) * 2 - 1

    def l_x_given_y(self, y, idx=None):
        """
        likehood function of x given y.
        idx == None is generate all the index.
        """
        def state_to_index(state):
            '''
            State space is in {1, -1}
            '''
            return (state + 1)/2

        if idx is None:
            idex = np.vstack([state_to_index(self.image), state_to_index(y)]).T
            query = np.array([self.p_filp[ix][iy] for ix, iy in idex])
        else:
            idex = np.vstack([state_to_index(self.image[idx]),
                              state_to_index(y[idx])]).T
            query = np.array([self.p_filp[ix][iy] for ix, iy in idex])
        return query

    def y_given_x(self, x):
        '''
        Target distribution, from baysian point of view.
        '''
        print 'This souldnt'
        pass

    def indexing(self, m, n):
        """
        2-D indexing to 1-D indexing
        """
        if(n >= 0 and m >= 0) and \
                m < self.image_shape[0] and n < self.image_shape[1]:

            return [self.image_shape[0] * n + m]
        else:
            return []

    def Energy_diff(self, y, x, idx=None, T=None):
        """
        Energy-based Distribution:
            This method calculate the energy difference of y and x
        If the idx argument is assigned then the function will
        output a same lenth of idx, and each element is the Energy-diff of
        the the corresponding idx, given the last_step image and next_step image
        """
        def id_to_loaction(idx):
            col = np.ceil(idx/self.image_shape[1])
            row = np.mod(idx, self.image_shape[1])
            return row, col

        diff = []
        if T is None:
            T = 1.
        if idx is None:
            for j in range(self.image_shape[1]):
                for i in range(self.image_shape[0]):
                    index = self.neiborhood(i, j)
                    diff += [(y[index].sum() * y[self.indexing(i, j)] -
                             x[index].sum() * x[self.indexing(i, j)])]
            out = np.exp(np.array(diff).ravel()/T)
        else:
            for i in list([idx]):
                row, col = id_to_loaction(i)
                index = self.neiborhood(row, col)
                diff += [(y[index].sum() * y[self.indexing(row, col)] -
                         x[index].sum() * x[self.indexing(row, col)])]
            out = np.exp(np.array(diff).ravel()/T)
        return out

    def neiborhood(self, m, n, r=1):
        """
        return the index of Neiborhoods
        In our case just r = 1
        """
        radius = [-1, 1]  # radius
        neiborhood_idx = []
        for x in radius:
            neiborhood_idx += sum([self.indexing(m + x, n)], [])
            neiborhood_idx += sum([self.indexing(m, n + x)], [])
        return neiborhood_idx

    def seqential_update(self, T=1, burn_in=1):
        last_step = self.init
        next_step = self.proposal_sample()
        for step in range(burn_in):
            for i in range(self.size[0]):
                accept = self.MH_accept(next_step, last_step, i, T=T)
                last_step[i] = self.switch(next_step, last_step, accept, i)[i]
                next_step = self.proposal_sample()
        return last_step

    def seqential_sample(self, n, T=1.):
        samples = []
        for i in range(n):
            samples += [self.seqential_update(T)]
        return np.vstack(samples)

    def tempure_gradient(self, y, x, T):
        return np.log(self.Energy_diff(y, x, T))/T

    def convolution_update(self):
        def select_block(i, M, N):
            block_ids = []
            for x in range(M):
                for y in range(N):
                    block_ids += [self.indexing(x, y)]
            return block_ids
        n_block = 100/10
        for block in range(n_block):
            id = select_block(block, 10, 10)
            self.init[id] = self.update()
        return self

    def gradient_boosted_updating(self, n_update=100, lr=0.1):
        last_step = self.init
        next_step = self.proposal_sample(last_step)
        T = np.ones(shape=self.size)
        mean_accept_rate = []
        for iter in xrange(n_update):
            accept = self.MH_accept(next_step, last_step, T)
            mean_accept_rate += [self.accept_rate(accept)]
            last_step = self.switch(next_step, last_step, accept)
            next_step = self.proposal_sample(last_step)
            T += lr * self.tempure_gradient(next_step, last_step, T)
            # gradient ascent
        return last_step


class seqential_sample:
    def __init__(self):
        pass


class gradient_estimate:
    def __init__(self, image, n_sample):
        self.n_sample = n_sample
        self.image = image.ravel()

    def sample_y_given_x(self, x):
        return Sampling_Image(x).sample(n=n_sample)

    def sample_x_given_y(self, y):
        def state_to_index(state):
            '''
            State space is in {1, -1}
            '''
            return (state + 1)/2

        p = self.p_filp[1][list(state_to_index(y))]
        return np.random.binomial(1, p=p,
                                  size=self.size) * 2 - 1

    def gibbs_sampling(self, k=1):
        '''
        the fucntion perform the k step of gibbs sampling
        '''
        y_given_x = self.sample_y_given_x(self.init)
        for i in range(k):
            x_given_y = self.sample_x_given_y(y_given_x)
            y_given_x = self.sample_y_given_x(x_given_y)
        return y_given_x

    def partial_energy(self, input):
        pass

    def gradient(self, k=1):
        return self.partial_energy(self.sample_y_given_x(
            self.image, n)).mean(0) - self.partial_energy(
                self.gibbs_sampling(n, k)).mean(0)

    def neiborhood(self, m, n, r=1):
        """
        return the index of Neiborhoods
        In our case just r = 1
        """
        radius = [-1, 1]  # radius
        neiborhood_idx = []
        for x in radius:
            neiborhood_idx += sum([self.indexing(m + x, n)], [])
            neiborhood_idx += sum([self.indexing(m, n + x)], [])
        return neiborhood_idx

    def indexing(self, m, n):
        """
        2-D indexing to 1-D indexing
        """
        if(n >= 0 and m >= 0) and \
                m < self.image_shape[0] and n < self.image_shape[1]:
            return [self.image_shape[0] * n + m]
        else:
            return []

    def update_temputure(self, T, lr=0.01, k=1):
        return T + lr * self.gradient(k)

    def gradient_boosted_updating(self, n_update=100, lr=0.1):
        last_step = self.init
        next_step = self.proposal_sample(last_step)
        T = np.ones(shape=self.size)
        mean_accept_rate = []
        for iter in xrange(n_update):
            accept = self.MH_accept(next_step, last_step, T)
            mean_accept_rate += [self.accept_rate(accept)]
            last_step = self.switch(next_step, last_step, accept)
            next_step = self.proposal_sample(last_step)
            T += lr * self.tempure_gradient(next_step, last_step, T)
            # gradient ascent
        return last_step


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # import cProfile
    f = open('../data/data4.txt')
    image = np.loadtxt(f)
    denoise = Denosing(image)
    last_step = denoise.image
    next_step = denoise.init
    cProfile.run("D = denoise.seqential_sample(10).mean(0).reshape(100, 100, order='F')")
    # D = denoise.sample(10).mean(0).reshape(100, 100)
    # D = denoise.gradient_boosted_updating(50, 2).reshape(100, 100, order='F')
    plt.matshow(D, cmap=plt.cm.gray)
    plt.show()
