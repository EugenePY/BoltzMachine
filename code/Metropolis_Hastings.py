# *-* coding: utf-8 *-*

import numpy as np


class MHasting(object):
    '''
    ############### Discrete Version ##############

    We want to sample a probability by a proposal distribution.
    Parameters:
        Sketch the algorithm:
            if r >= 1, x accpted x, otherwise
        Targrt dis: pi
        proposal:
    '''

    def __init__(self, target, n_vis=1, n_sample=1, rng=None):
        '''
        pi: a random stream object (state,) map random state to probability
        (pdf / pmf). It's a mapping.
        '''
        self.pi = target
        self.size = (n_sample, n_vis)

        if rng is None:
            self.rng = np.random.RandomState()

        self.init = self.rng.binomial(1, p=0.5, size=self.size)

    def proposal_sample(self, x):
        '''
        can be override by inherent.
        '''
        return self.rng.binomial(1, p=0.5, size=self.size)

    def r(self, y, x):
        return self.pi(y)/self.pi(x)

    def MH_accept(self, y, x):
        r = self.r(y, x)
        next_accept = r > self.rng.uniform(size=self.size)
        accept = np.vstack([(1-next_accept).astype(bool), next_accept])
        return accept

    def switch(self, y, x, accept, idx=None):
        X = np.vstack([x, y]).T
        return X[accept.T]

    def accept_rate(self, accept):
        return accept[-1].mean()

    def sample(self, n, burn_in=10, T=.01):
        samples = []
        for i in range(n):
            last_step = self.init
            next_step = self.proposal_sample(last_step)
            mean_accept_rate = []
            for step in xrange(burn_in):
                accept = self.MH_accept(next_step, last_step)
                mean_accept_rate += [self.accept_rate(accept)]
                last_step = self.switch(next_step, last_step, accept)
                next_step = self.proposal_sample(last_step)
            print "Accept rate: %.4f" % (np.mean(mean_accept_rate))
            samples += [last_step]

        return np.vstack(samples)


class MH_normal(MHasting):
    """
    ############## Continous Version ##############

    """
    def __init__(self, target, n_sample):
        MHasting.__init__(self, target, n_sample)
        self.init = self.rng.normal(size=self.size)

    def proposal_sample(self, x):
        return self.rng.normal(loc=x, size=self.size)

if __name__ == "__main__":
    def ber(x):
        event = x
        return (0.7**event)*(0.3**(1-event))

    def double_exp(x):
        return np.exp(-abs(x))*0.5

    model = MH_normal(double_exp, n_sample=1000)
    data = model.sample()
    print data.mean()
    print data.var()
    P = double_exp(np.random.laplace(loc=0, size=(1000,)))
    likelihood_dis = double_exp(data)
    print "KL divergence:%.4f" % ((P * np.log(P) - P *
                                   np.log(np.array(likelihood_dis))).mean())
