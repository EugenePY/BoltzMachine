import theano
import theano.tensor as T
import theano.sparse as sp
import cPickle as pk
import numpy as np

from ParallelTempering import ParallelTempering
from hidden_layer import HiddenLayer


class LayerStacking(object):

    __stacked = 0

    def __init__(self):
        self.n_layers = []

    def __str__(self):
        output = 'LayerStacking\n'
        output += "Layer Structure\n"
        output += "%i HiddenLayers\n" % (self.n_layers)
        for idx, LayerObject in enumerate(zip(self.layers,
                                              self.downpass_layers)):
            output += "Layer %i: (%i vis units, %i hid units)," % (
                idx, LayerObject[0].n_visible, LayerObject[0].n_hidden)
            output += "DownPassLayer %i: (%i vis units, %i hid units)\n" % (
                idx, LayerObject[1].n_hidden, LayerObject[1].n_visible)
        return output

    def getitem(self):
        return self.__dict__

    def stack(self):
        if self.__stacked is 0:
            self.layers = []
            self.rbm_layers = []
            self.downpass_layers = []

            for i in range(self.n_layers):
                # let the output of each layer become the input of
                # the next layer.
                if i == 0:
                    input = self.input
                    n_visible = self.n_visible
                else:
                    input = self.layers[-1].output
                    n_visible = self.n_hidden[i-1]

                n_hidden = self.n_hidden[i]
                self.rbm_layers.append(ParallelTempering(
                    input=input, n_visible=n_visible, n_hidden=n_hidden)
                                    )
                self.layers.append(HiddenLayer(input=input, n_in=n_visible,
                                               n_out=n_hidden,
                                               W=self.rbm_layers[i].W,
                                               vbias=self.rbm_layers[i].vbias,
                                               hbias=self.rbm_layers[i].hbias
                                               )
                                   )
                print "\rFowardProbagation Layer: %i Done..." % (i)

            # building the downpass network.
            for i in reversed(xrange(self.n_layers)):
                if i == (self.n_layers - 1):
                    # last layer is same with feedfoward
                    self.downpass_layers.append(self.layers[i])
                elif i == (self.n_layers - 2):
                    # Change the layer
                    # The layer configration is a fliped hidden layer object
                    # Take the output from the last layer(except for
                    # last 2 layers.)
                    self.downpass_layers.append(HiddenLayer(
                        input=self.downpass_layers[-1].feedbackward,
                        n_in=self.downpass_layers[-1].n_visible,
                        n_out=self.rbm_layers[i].n_visible,
                        W=self.rbm_layers[i].W.T,
                        vbias=self.rbm_layers[i].hbias,
                        hbias=self.rbm_layers[i].vbias
                    ))
                else:
                    self.downpass_layers.append(HiddenLayer(
                        input=self.downpass_layers[-1].output,
                        n_in=self.downpass_layers[-1].n_hidden,
                        n_out=self.rbm_layers[i].n_visible,
                        W=self.rbm_layers[i].W.T,
                        vbias=self.rbm_layers[i].hbias,
                        hbias=self.rbm_layers[i].vbias
                    ))
                print 'BackProbagation Layer: %i Done...' % (i)
            print 'Stacking Done'
            self.__stacked = 1
        else:
            print "The Network has been builded"
        return self


class PTDBN(LayerStacking):

    def __init__(self, input, n_layers, n_visible, n_hidden, batch_size,
                 rbm_layers=None):
        '''
        A DBN is formed by a RBM on the top and sigmoid belief net(other)
        layers.

        '''
        if rbm_layers is not None:
            pass

        LayerStacking.__init__(self)
        self.input = input
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.batch_size = batch_size
        self.n_layers = len(n_hidden)
        self.stack()

    def reconstruction(self, i):
        for layer in self.layers:
            pass

    def build_greedy_train_fn(self, x, train_set_x, T0=0.85, lr=1,
                              mom=0.9, l2=0., k=5, l1=0., rbm_layers=None):
        # To record the hyper-parameters
        self.T0 = T0
        self.l1 = l1
        self.l2 = l2
        greedy_train_fn = []
        idx = T.iscalar('idx')
        print 'Build Stacking Graph'
        for i in range(self.n_layers):
            if rbm_layers is not None:
                for i, layers in enumerate(rbm_layers):
                    self.rbm_layers[i].W, self.rbm_layers[i].hbias,
                    self.rbm_layers[i].vbias = layers.W, layers.hbias, \
                        layers.vbias

            persistent = theano.shared(
                np.zeros((k, self.batch_size, self.rbm_layers[i].n_hidden)
                         ).astype(theano.config.floatX), borrow=True)

            cost, accept_rate, updates = \
                self.rbm_layers[i].PT_get_cost_update(
                    persistent=persistent, T0=T0, lr=lr, k=k, l2=l2, l1=l1)

            train = theano.function([idx], cost, updates=updates,
                                    givens={x:
                                            train_set_x[idx * (self.batch_size):
                                                        (idx + 1) *
                                                        self.batch_size]
                                            })
            accept_rate = theano.function([idx],
                                          accept_rate, updates=updates,
                                          givens={
                                              x:
                                              train_set_x[idx *
                                                          self.batch_size:
                                                          (idx+1) *
                                                          self.batch_size]
                                          })

            greedy_train_fn.append((train, accept_rate, persistent))

        print 'Building Done...'
        return greedy_train_fn

    def training(self, x, train_set_x, gamma, T0=0.9, lr=1., l2=0., l1=0.,
                 batch_size=200, n_epoch=15, rbm_layers=None, k=6):

        self.fns = self.build_greedy_train_fn(x, train_set_x, T0, lr,
                                              gamma, l2, k, l1, rbm_layers)
        n_batch = train_set_x.get_value(borrow=True).shape[0]/batch_size
        print "Start Training"
        for key, (train, accept_rate, persistent) in enumerate(self.fns):
            his_cost = []

            for i in range(n_epoch):
                    epoch_cost = []
                    epoch_accept_rate = []
                    if i <= 10:
                        # momen = 0.5
                        pass
                    else:
                        # momen = 0.9
                        pass
                    for index in range(n_batch):
                        epoch_cost += [train(index)]
                        epoch_accept_rate = accept_rate(index)
                        if (index % 10) == 0:
                            print np.mean(persistent.get_value(borrow=True),
                                          axis=(1, 2))
                            print 'Layer %i Peeking cost:%.3f AR:' % \
                                (key, np.mean(epoch_cost),), epoch_accept_rate
                        if (not bool(np.isnan(np.mean(epoch_cost))) and
                                not (index % 1000 != 0)) is True:
                            pk.dump({'model': self.getitem(),
                                     'cost': his_cost},
                                    open('./PTDBM.layer.1.p', 'wb'), -1)
                        elif bool(np.isnan(np.mean(epoch_cost))) is True:
                            print 'Nan detacted'
                            raise OverflowError
                    print "Layer %i Traing Epoch %i pesudo_likehood: %.3f" % \
                        (key, i, np.mean(epoch_cost))
                    his_cost += [np.mean(epoch_cost)]

    def save(self):
        return (self.__class__, self.__dict__)

    def load(self, attributes):
        self.__dict__.update(attributes)
        return self


if __name__ == '__main__':
    train_set = pk.load(open('/home/tn00372136/nlp-lab/Rec/data_lazy_test/\
penguin.train.rbm.debug.csc.p', 'rb')).astype(theano.config.floatX)
    fitted = pk.load(open('./PTDBM.layer.p', 'rb'))['model'][-1][0]
    shu = pk.load(open('./index', 'rb'))
    shu[:np.ceil(train_set.shape[0]*0.7)]
    train_set = train_set[shu]
    train_set = theano.shared(train_set, borrow=True)
    n_vis = train_set.get_value(borrow=True).shape[1]
    x = sp.csc_matrix('x')
    dens_x = x.toarray()
# testing the graph building process
    model = PTDBN(input=dens_x, n_layers=2, n_hidden=[200, 100],
                  n_visible=n_vis, batch_size=100)
    model.training(x, train_set_x=train_set, gamma=0.9)
    pk.dump(model.save(), open('./PTDBN.p', 'wb'), -1)
