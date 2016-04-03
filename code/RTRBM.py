# -*-coding=utf-8-*-
import numpy as np
from rbm import RBM
import cPickle as pk

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams


class ShiftedRBM(RBM):
    '''
    Building block of the Recurrent Temporal RBM
    Create "Conditional" inference method.
    '''
    def __init__(self, input, n_hid, n_vis, Wp=None,
                 W=None, hbias=None, vbias=None):
        # build parameters of Shifted RBM
        RBM.__init__(self, input, n_visible=n_vis, n_hidden=n_hid,
                     W=W, hbias=hbias, vbias=vbias)

    def free_energy_given_hid_lag(self, v_sample, Wp, hid_lag):
        wx_b = T.dot(v_sample, self.W) + T.dot(hid_lag, Wp) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup_given_hid_lag(self, vis, Wp, hid_lag):
        pre_sigmoid_activation = T.dot(vis, self.W) + T.dot(hid_lag, Wp) + \
            self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v_hid_lag(self, v0_sample, Wp, hid_lag):
        pre_activation_h1, h1_mean = self.propup_given_hid_lag(v0_sample, Wp,
                                                               hid_lag)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_activation_h1, h1_mean, h1_sample]

    def gibbs_vhv_given_h_lag(self, v0, Wp, h_lag):
        pre_activation_h1, h1_mean, h1_sample = self.sample_h_given_v_hid_lag(
            v0, Wp, h_lag)
        pre_activation_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_activation_h1, h1_mean, h1_sample,
                pre_activation_v1, v1_mean, v1_sample]
    # other Inference method is the same


class RTRBM(RBM):
    '''
    From the paper,.
    Theano automatically doing backpropgation through time.
    '''
    def __init__(self, input, n_hid, n_vis, time, h0=None, Wp=None,
                 vbias=None, hbias=None, W=None, activation=None):
        self.input = input
        self.n_hid = n_hid
        self.n_vis = n_vis
        self.T = time
        # parameters
        if activation is None:
            activation = T.nnet

        if h0 is None:
            h0 = theano.shared(np.zeros(self.n_hid, dtype=theano.config.floatX),
                               borrow=True, name='h0')

        if Wp is None:
            Wp = theano.shared(
                np.random.normal(size=(self.n_hid, self.n_hid)).astype(
                    theano.config.floatX), borrow=True, name='Wp')

        if W is None:
            W = theano.shared(
                np.random.normal(size=(self.n_vis, self.n_hid)).astype(
                    theano.config.floatX), borrow=True, name='W')

        if vbias is None:
            vbias = theano.shared(
                np.zeros(self.n_vis, dtype=theano.config.floatX),
                borrow=True, name='vbias')

        if hbias is None:
            hbias = theano.shared(
                np.zeros((self.n_hid), dtype=theano.config.floatX),
                borrow=True, name='hbias')

        self.activation = activation
        self.h0 = h0
        self.Wp = Wp
        self.W = W
        self.vbias = vbias
        self.hbias = hbias
        self.params = [self.h0, self.Wp, self.W, self.vbias, self.hbias]
        # Create the RTRBM network
        # Construct RBMs network which sharing the weights.
        self.temporal_layers = []
        for t in range(self.T):
            self.temporal_layers += [ShiftedRBM(
                self.input[:, t*self.n_vis:(t+1) * self.n_vis],
                n_vis=self.n_vis,
                n_hid=self.n_hid,
                W=self.W,
                vbias=self.vbias,
                hbias=self.hbias)]

    def one_V_temproal_sampling(self, V):
        V_sample = []
        H = [self.h0]
        # updated the bias term
        for t, layer in enumerate(self.temporal_layers):
            pre_sigmoid_h1, h1_mean, h1_sample, \
                pre_sigmoid_v1, v1_mean, v1_sample = (
                    layer.gibbs_vhv_given_h_lag(V[:, t * self.n_vis:(t+1) *
                                                  self.n_vis], self.Wp, H[-1])
                                                      )
            V_sample += [v1_sample]
            H += [h1_mean]
        return T.concatenate(V_sample, axis=1)

    def H_given_V(self, V, Wp, h0):
        H = [h0]
        for t, layer in enumerate(self.temporal_layers):
            H += [layer.propup_given_hid_lag(V[:,
                                               t*self.n_vis:(t+1)*self.n_vis],
                  Wp, H[-1])[1]]
        return H[1:]

    def free_energy_RTRBM(self, V):
        H = self.H_given_V(V, self.Wp, self.h0)
        free_energy = []
        # This part is the hot spot because we comput the free energy seperately
        for t in range(self.T):
            free_energy += [self.temporal_layers[t].free_energy_given_hid_lag(
                V[:, t*self.n_vis:(t+1)*self.n_vis], self.Wp,
                H[t])]
        return sum(free_energy)

    def get_cost_updates(self, lr=0.01, k=1, PCD=None, PT=None):
        chain_start = self.input

        V_burn_in, updates = theano.scan(fn=self.one_V_temproal_sampling,
                                         outputs_info=[chain_start],
                                         n_steps=k,
                                         name='RTRBM Gibbs Sampler')
        V_sample = V_burn_in[-1]
        KL_diff = self.free_energy_RTRBM(self.input) - \
            self.free_energy_RTRBM(V_sample)
        # This part is the hot spot because we comput the free energy seperately

        KL_diff = T.mean(KL_diff)
        self.gparams = T.grad(KL_diff, self.params,
                              consider_constant=[V_sample])
        for param, gparam in zip(self.params, self.gparams):
            updates[param] = param - lr*gparam

        cost, updates = self.get_pseudo_likelihood_cost(updates)

        return cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.free_energy_RTRBM(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy_RTRBM(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_vis * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                        fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_vis
        return cost, updates


class MultiRBM(RBM):
    def __init__(self, input, n_vis, n_hid, n_cate,
                 W=None, vbias=None, hbias=None):
        '''
        The input should be a 3D tensor with (n_cat, N_sample, n_vis)
        '''
        self.input = input
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.n_cate = n_cate

        if W is None:
            W = theano.shared(np.random.normal(size=(self.n_cate, self.n_vis,
                                                     self.n_hid)).astype(
                                                    theano.config.floatX),
                              borrow=True)
        if vbias is None:
            vbias = theano.shared(np.zeros(
                shape=(self.n_cate, self.n_vis,)).astype(theano.config.floatX),
                borrow=True)

        if hbias is None:
            hbias = theano.shared(
                np.zeros(
                    shape=(self.n_hid,)).astype(
                        theano.config.floatX),
                borrow=True)
        self.numpy_rng = np.random.RandomState(1234)
        self.theano_rng = MRG_RandomStreams(self.numpy_rng.randint(2 ** 30))
        self.W = W
        self.vbias = vbias
        self.hbias = hbias

    def free_energy(self, vis):
        vW_b = T.batched_dot(vis, self.W) + T.addbroadcast(self.hbias, 1)
        visible_term = T.batched_dot(vis, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(vW_b)), axis=2)
        return T.sum(-hidden_term-visible_term, axis=0)

    def propup(self, vis):
        x = T.batched_dot(vis, self.W) + self.hbias
        return [x, T.nnet.sigmoid(x)]

    def propdown(self, hid):
        x = T.batched_dot(hid, self.W.dimshuffle(0, 2, 1)) + \
            self.vbias.dimshuffle((0, 'x', 1))
        e_x = T.exp(x - x.max(axis=0, keepdims=True))
        out = e_x / e_x.sum(axis=0, keepdims=True)
        return [x, out]

    def sample_v_given_h(self, hid):
        x, out = self.propdown(hid)
        v_sample = []
        for v in range(self.n_vis):
            v_sample += [self.theano_rng.multinomial(
                n=1, pvals=out[:, :, v].T).dimshuffle(1, 0, 'x')]
        v_sample = T.concate(v_sample, axis=2)
        return [x, out, v_sample]

    def sample_h_given_v(self, vis):
        x, out = self.propup(vis)
        h_sample = self.theano_rng.binomial(n=1, p=out, size=out.shape)
        return [x, out, h_sample]


class DS_MRTRBM:
    def __init__(self, input, n_vis, time, batch_size, n_cate):
        # the input dimeansion should be (n_cate, N_sample * time * n_vis)
        self.input = input
        self.n_vis = n_vis
        self.time = time
        self.batch_size = batch_size
        self.n_cate = n_cate
        self.n_batch = self.input.get_value(borrow=True).shape[1] / \
            self.batch_size

    def mini_batch_formated(self, t):
        index = self.batch_size * self.time * self.n_vis
        temp = self.input[:, t*index:(t+1)*index].toarray().reshape(
            (self.n_cate, self.batch_size, self.time, self.n_vis))
        return temp.dimshuffle(2, 0, 1, 3)


class MultiRTRBM(DS_MRTRBM):
    """This Class Implement the Multi-Category Recurrent Temporal RBM """
    def __init__(self, input, n_visible, n_hidden, time, n_cate,
                 W=None, Wt=None, vbias=None, hbias=None, h0=None):
        # the input dimeansion should be (n_cate, N_sample * time * n_vis)
        self.input = input
        self.n_vis = n_visible
        self.n_hid = n_hidden
        self.time = time
        self.n_cate = n_cate
        # Define the parameter of the Machine
        if W is None:
            W = theano.shared(np.random.normal(
                size=(self.n_cate, self.n_vis, self.n_hid)).astype(
                    theano.config.floatX))

        if vbias is None:
            vbias = theano.shared(
                np.zeros(shape=(self.n_cate, self.time, self.n_vis)).astype(
                    theano.config.floatX))

        if hbias is None:
            hbias = theano.shared(
                np.zeros(shape=(self.time, self.n_hid)).astype(
                    theano.config.floatX))

        if Wt is None:
            Wt = theano.shared(
                np.random.normal(size=(self.n_hid, self.n_hid)).astype(
                    theano.config.floatX))

        if h0 is None:
            h0 = theano.shared(
                np.zeros(shape=(1, 1, self.n_hid)).astype(theano.config.floatX))
        # set parameters
        self.W = W
        self.Wt = Wt
        self.h0 = h0
        self.hbias = hbias
        self.vbias = vbias
        self.params = [self.W, self.Wt, self.h0, self.hbias, self.vbias]
        self.numpy_rng = np.random.RandomState(1234)
        self.theano_rng = MRG_RandomStreams(self.numpy_rng.randint(2 ** 30))

    def h_given_h_lag_vt(self, vt, h_lag, hbias):
        if h_lag == self.h0:
            x = T.batched_dot(vt, self.W) + T.addbroadcast(
                T.dot(h_lag, self.Wt) + hbias.dimshuffle('x', 0), 0, 1)
        else:
            x = T.batched_dot(vt, self.W) + \
                T.dot(h_lag, self.Wt) + hbias.dimshuffle('x', 0)
        return [x, T.nnet.sigmoid(x)]

    def H_given_h_lag_vt(self, V):
        H = [self.h0]

        # [x, out], _ = theano.scan(fn=self.h_given_h_lag_vt, sequence=V,
        #                           outputs_info=[None, self.h0],
        #                          n_steps=V.shape[0])
        for t in range(self.time):
            H += [self.h_given_h_lag_vt(V[t], H[-1], self.hbias[t])[1]]
        return T.concatenate(H[1:], axis=2)

    def free_energy_given_hid_lag(self, vt, h_lag, hbias, vbias):
        if h_lag == self.h0:
            wx_b = T.batched_dot(vt, self.W) +\
                T.addbroadcast(T.dot(h_lag, self.Wt) + hbias, 0, 1)
            vbias_term = T.batched_dot(vt, vbias)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=2)
        else:
            wx_b = T.batched_dot(vt, self.W) + T.dot(h_lag, self.Wt) + \
                hbias.dimshuffle('x', 0)
            vbias_term = T.batched_dot(vt, vbias)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=2)
        return -hidden_term - vbias_term

    def free_energy_RTRBM(self, V):
        H = self.H_given_h_lag_vt(V)
        for t in range(self.time):
            if t == 0:
                Et = T.sum(self.free_energy_given_hid_lag(
                    V[t], self.h0, self.hbias[t], self.vbias[:, t, :]), axis=0)
            else:
                Et += T.sum(self.free_energy_given_hid_lag(
                    V[t], H[:, :, t*(self.n_hid):(t+1)*self.n_hid],
                    self.hbias[t], self.vbias[:, t, :]), axis=0)
        return Et

    def propup_given_h_lag(self, vt, h_lag, hbias):
        if h_lag == self.h0:
            x = T.batched_dot(vt, self.W) + T.addbroadcast(
                T.dot(h_lag, self.Wt) + hbias, 0, 1)
        else:
            x = T.batched_dot(vt, self.W) + hbias + T.dot(h_lag, self.Wt)
        return [x, T.nnet.sigmoid(x)]

    def propdown_given_h_lag(self, ht, vbias):
        x = T.batched_dot(ht, self.W.dimshuffle(0, 2, 1)) + \
            vbias.dimshuffle((0, 'x', 1))
        e_x = T.exp(x - x.max(axis=0, keepdims=True))
        out = e_x / e_x.sum(axis=0, keepdims=True)
        return [x, out]

    def sample_vt_given_ht_h_lag(self, ht, vbias):
        x, out = self.propdown_given_h_lag(ht, vbias)
        v_sample = []
        for v in range(self.n_vis):
            v_sample += [self.theano_rng.multinomial(
                n=1, pvals=out[:, :, v].T,
                dtype=theano.config.floatX).dimshuffle(1, 0, 'x')]
        v_sample = T.concatenate(v_sample, axis=2)
        return [x, out, v_sample]

    def sample_ht_given_vt_hid_lag(self, vt, h_lag, hbias):
        x, out = self.propup_given_h_lag(vt, h_lag, hbias)
        h_sample = self.theano_rng.binomial(n=1, p=out, size=out.shape,
                                            dtype=theano.config.floatX)
        return [x, out, h_sample]

    def gibbs_vhv_given_h_lag(self, v0, h_lag, hbias, vbias):
        xh, ph, h0 = self.sample_ht_given_vt_hid_lag(v0, h_lag, hbias)
        xv, pv, v1 = self.sample_vt_given_ht_h_lag(h0, vbias)
        return [xh, ph, h0, xv, pv, v1]

    def gibbs_VhV(self, V0):
        V = []
        H = self.H_given_h_lag_vt(V0)
        for t in range(self.time):
            if t == 0:
                V += [self.gibbs_vhv_given_h_lag(
                    V0[t], self.h0, self.hbias[t],
                    self.vbias[:, t, :])[-1].dimshuffle('x', 0, 1, 2)]
            else:
                V += [self.gibbs_vhv_given_h_lag(
                    V0[t], H[:, :, t*self.n_hid:(t+1)*self.n_hid],
                    self.hbias[t],
                    self.vbias[:, t, :])[-1].dimshuffle('x', 0, 1, 2)]
        return T.concatenate(V, axis=0)

    def get_cost_updates(self, persistant, k=2, lr=0.01, l1=0., l2=0.01):
        chain_start = persistant
        V_burn_in, updates = theano.scan(fn=self.gibbs_VhV,
                                         outputs_info=[chain_start],
                                         n_steps=k,
                                         name='MultiRTRBM Gibbs Smapler')

        chain_end = V_burn_in[-1]
        # Contrastive Divergence (Variational method Cost)/ Approxiamted
        # likelihood
        L1 = T.sum(T.abs_(self.W)) + T.sum(T.abs_(self.Wt))
        L2 = T.sum(self.W**2) + T.sum(self.Wt**2)
        KL_diff = T.mean(self.free_energy_RTRBM(self.input) -
                         self.free_energy_RTRBM(chain_end)) +\
            T.cast(l1, theano.config.floatX) * L1 + \
            T.cast(l2, theano.config.floatX) * L2
        self.gparams = T.grad(KL_diff, self.params,
                              consider_constant=[chain_end])
        for param, gparam in zip(self.params, self.gparams):
            if param in [self.W, self.Wt]:
                updates[param] = param - 0.0001 * gparam
            else:
                updates[param] = param - lr * gparam
        cost, updates = self.get_pseudo_likelihood_cost(updates)

        return cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.free_energy_RTRBM(xi)
        for k in range(self.n_cate):
            xi_flip = T.set_subtensor(xi[:, k, :, bit_i_idx],
                                      1 - xi[:, k, :, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy_RTRBM(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_vis * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                        fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_vis
        return cost, updates


def main():
    corpus = pk.load(open('./Traning.ptt.no.window.p', 'rb')
                     ).values()[0].astype(theano.config.floatX)
    data = theano.shared(corpus, borrow=True)
    index = T.iscalar('index')
    pre_process = DS_MRTRBM(input=data,
                            n_cate=data.get_value(borrow=True).shape[0],
                            time=3, batch_size=1000, n_vis=3
                            )
    n_vis = pre_process.n_vis
    n_hid = 4
    time = pre_process.time
    n_cate = pre_process.n_cate
    TESTING_DATA = pre_process.mini_batch_formated(index)
    model = MultiRTRBM(input=TESTING_DATA, n_visible=n_vis, n_hidden=n_hid,
                       n_cate=n_cate,
                       time=time)
    per = model.input.copy()
    cost, updates = model.get_cost_updates(per)
    print 'Building The Graph'
    training = theano.function([index], cost, updates=updates)
    print 'Building Done'
    H = []
    for epoch in range(20):
        hist_cost = []
        for idex in range(100):
            hist_cost += [training(idex)]
        print 'Training epoch %i pesudo likehood %.3f' % (epoch,
                                                          np.mean(hist_cost))
        H += [np.mean(hist_cost)]
        pk.dump({'model': model, 'hist_cost': H},
                open('MultiRTRBM.model.p', 'wb'), -1)

if __name__ == '__main__':
    main()
