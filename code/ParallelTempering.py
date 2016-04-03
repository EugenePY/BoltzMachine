import theano
import numpy as np

import theano.tensor as T
from collections import OrderedDict
from rbm import RBM


class MCMC(RBM):
    def __init__(self, input, n_visible, n_hidden):
        RBM.__init__(self, input=input, n_visible=n_visible,
                     n_hidden=n_hidden)


class ParallelTempering:
    def __init__(self, MCMC):
        """
        input object containing a well define gibbs chain,
        propup, propdown
        """
        self.model = MCMC

    def important_sampling(self):
        '''
        Use free energy for as the importance.
        '''
        pass

    def free_energy_vi(self, v):
        pass

    def outputs_switch(self, last_step, next_step,
                       last_temp, next_temp):
        '''
        Switch funciton representing the switch beteen x, y
        T.switch(cond, if True, if False);
        '''
        accept, accept_rate = self.replica_switch(last_step, next_step,
                                                  last_temp, next_temp)
        return T.switch(accept, next_step, last_step)

    def replica_switch(self, last_step, next_step,
                       last_temp, next_temp):
        """
        T.matrix last_step, T.matrix next_step, T.scalar last_temp,
        T.scalar next_step
        """
        r = (self.model.free_energy(last_step) -
             self.model.free_energy(next_step)) * (last_temp - next_temp)
        accept = T.ge(r.dimshuffle(0, 'x'),
                      self.model.theano_rng.uniform(size=last_step.shape))
        accpet_rate = T.mean(accept)
        return accept, accpet_rate

    def propup_temp(self, vis, temp=1):
        """
        Coresspoinding to the propup function
        """
        pre_sigmoid = (self.model.propup(vis)[0]) *\
            T.cast(temp, dtype=theano.config.floatX)
        return [pre_sigmoid, T.nnet.sigmoid(pre_sigmoid)]

    def propdown_temp(self, hid, temp=1):
        """
        temped propdown
        """
        pre_sigmoid = (self.model.propdown(hid)[0]) *\
            T.cast(temp, dtype=theano.config.floatX)
        return [pre_sigmoid, T.nnet.sigmoid(pre_sigmoid)]

    def sample_v_given_h_temp(self, h0_sample, temp=1):
        pre_sigmoid_v1, v1_mean = self.propdown_temp(h0_sample, temp)
        v1_sample = self.model.theano_rng.binomial(size=v1_mean.shape, n=1,
                                                   p=v1_mean,
                                                   dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def sample_h_given_v_temp(self, v0_sample, temp=1):
        pre_sigmoid_h1, h1_mean = self.propup_temp(v0_sample, temp)
        h1_sample = self.model.theano_rng.binomial(size=h1_mean.shape, n=1,
                                                   p=h1_mean,
                                                   dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv_temp(self, temp, v0_sample):
        """
        The temp argument must be the first one due to SCAN fucntion
        restriction.
        """
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v_temp(
            v0_sample, temp)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h_temp(
            h1_sample, temp)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh_temp(self, h0_sample, temp=1):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h_temp(
            h0_sample, temp)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v_temp(
            v1_sample, temp)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def PCD_tempering(self, chain_start, persistent_temp, T0, k=3):
        '''
        This method perform one step gibbs sampling, and K tempering.
        each scan loop will create 1 step gibbs sampling step with
        temperture_{i}
        We use SCAN function to generate the parallel tempertured visable unit.
        Then use SCAN function scan through the temped visable units and perform
        the update.
        '''
        chain_start = chain_start
        temp_space = theano.shared(np.linspace(1, T0,
                                               k).astype(theano.config.floatX))
        (
            [
                pre_sigmoid_nhs,
                nh_means,
                nh_samples,
                pre_sigmoid_nvs,
                nv_means,
                nv_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_vhv_temp,
            outputs_info=[None, None, None, None, None, None],
            non_sequences=chain_start,
            sequences=temp_space
        )
        return nv_samples, temp_space, updates

    def parallel_update(self, chain_start, persistent, T0=0.1, k=3):
        """
        TODO:
        """
        k_temper, temp_sapce, updates = self.PCD_tempering(
            chain_start, T0=T0, persistent_temp=persistent, k=k)
        # we have create the k temper using one step PCD with differernt
        # temputure
        persistent = k_temper
        for i in reversed(xrange(1, k)):
            persistent = T.set_subtensor(persistent[i-1],
                                         self.outputs_switch(k_temper[i],
                                         k_temper[i-1],
                                         temp_sapce[i],
                                         temp_sapce[i-1]))
        return persistent, updates


class MetroplisedGibbs:
    def __init__(self):
        pass


if __name__ == '__main__':
    TEST_DATA = theano.shared(np.random.binomial(
        1, p=0.5, size=(100, 10)).astype(theano.config.floatX))
    persistent = theano.shared(
        np.zeros(shape=(3, 100, 10)).astype(theano.config.floatX))
    PT = ParallelTempering(MCMC(input=TEST_DATA,
                           n_visible=10,
                           n_hidden=10))
    nv_samples, updates = PT.parallel_update(TEST_DATA, persistent, T0=0.01)
    print 'Building the Computional Graph'
    fn = theano.function([], nv_samples, updates=updates)
    print 'Building Done'
    print fn()
