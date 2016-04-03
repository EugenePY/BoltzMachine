import numpy as np


class People(object):
    '''
    This class is just for variable transformation (A complex one.....), compare
    to X_{i}>100 (gien 100 < k), X_{i} iid binominal(p, k).
    Game rules
    '''
    pass


class Intergral_approximation(object):
    '''
    The power of law of large number v.s Taylor expansion ??
    By taylor expansion, we know that the law of large number is assured by the
    existance of second moment.
    '''
    def __init__(self, size):
        # this is N by K(N is sample size K is the number of variables)
        self.size = size
        self.init = np.random.normal(size=self.size[0])

    def sample_k1_given_k0(self, x_k0, k):
        return np.random.normal(loc=x_k0**2, scale=np.sqrt(k/2.),
                                size=self.size[0])

    def foward_sample_all_given_0(self, x0):
        sample = self.sample_k1_given_k0(x0, 1)
        for k in range(2, self.size[1]):
            sample = np.vstack([sample, self.sample_k1_given_k0(x0, k)])
        return sample

    def backward_sample_0_given_all(self, all):
        '''
        input should
        '''
        return self.sample_k1_given_k0(all[0], 1)

    def gibbs_sampling(self, k_step=10):
        sample = self.init
        for i in range(k_step):
            sample = self.backward_sample_0_given_all(
                self.foward_sample_all_given_0(sample))
        return np.vstack([sample, self.backward_sample_0_given_all(
                self.foward_sample_all_given_0(sample))])

    def sample(self):
        pass


class rare_event:
    def __init__(self):
        self.budget = 2.
        self.bet = 0.06
        self.init = 20.
        self.k_games = 400.
        self.n_sample = 100000
        self.shift_mean = (200.-20)/self.k_games + 0.06
        self.game_hist = self.weighted_samping()
        self.debt = 200

    def estimate_p(self):
        print 'The transformed Space distribution %.3f' % \
            self.variable_trans(self.game_hist).mean()
        return (self.variable_trans(self.game_hist) * self.recovery()).mean()

    def variable_trans(self, game_hist):
        current_budget = np.cumsum(game_hist, 1) + self.init
        states = (current_budget < 0).sum(1)
        states = (states == 0)
        states[current_budget[:, -1] < self.debt] = False
        return states

    def recovery(self):
        return np.prod(np.exp(1./2*(self.shift_mean-self.bet)**2 -
                              self.game_hist * (self.shift_mean - self.bet)), 1)

    def weighted_samping(self):
        samples = np.random.normal(loc=self.shift_mean - self.bet,
                                   size=(self.n_sample, self.k_games))
        samples[:, 0] = samples[:, 0] + self.budget
        return samples


class CMonte_carlo(object):
    '''
    Homework 10 solution. Imagine your are in the casino and you
    have to win 200,000,000 today, otherwise you get filled. Can you estimate
    the probability of you get the 200,000,000 peacefully. You have 20,000,000
    as your money to lay you bet. The idea is to use conditional prbability.
    In this class we use important sampling technique to estimate the Rare
    event.

    The MC method estimate the X_{i} follow the Binominal(p, k). Which p is
    very small.

    '''

    def __init__(self, G_win=21./38.):
        '''
        Meta parameter set up
        '''
        self.debt = 200.
        self.init = 20.
        self.P_win = 18./38.

        self.bet = 1.

        self.G_win = np.sqrt(200./400)
        self.generate_rand(1000, 400)

    def variable_trans(self, game_hist):
        current_budget = np.cumsum(game_hist * 2 - 1, 1) + self.init
        # create the income streams
        print current_budget
        current_budget = current_budget + self.init
        states = (current_budget < 0).sum(1)  # delet the bankcrupt stream
        states = (states == 0)  # keep the not bankcrupt streams
        states[current_budget[:, -1] < self.debt] = 0  # remove below 200
        return states

    def generate_rand(self, n_sample, k_games):
        self.game_his = np.random.binomial(1, self.G_win, size=(n_sample,
                                           k_games))
        return self.game_his

    def recovery(self):
        weighted = self.P_win/self.G_win
        weighted_lost = (1 - self.P_win)/(1 - self.G_win)
        factor_dis = (weighted)**self.game_his * \
                     (weighted_lost)**(1 - self.game_his)
        return np.prod(factor_dis, axis=1)

    def estimate(self):
        print self.variable_trans(self.game_his).mean()
        return np.mean(self.variable_trans(self.game_his)*self.recovery())


if __name__ == '__main__':
    print 'First the G_win = 20/38 estimate the expection of \
binomial(p=18/38, k=300)'
    print 'Good Approximation:(Estimate:%lf, True Value:%.3f)\n'
    print CMonte_carlo().estimate()
    print 'let G_win = 25/38 estimate the expection of \
binomial(p=18/38, k=300)'
    print 'Bas Approximation:(Estimate:%.3f, True Value:%.3f)' % \
        (CMonte_carlo(25./38.).estimate(), 18./38.*300)

    # The problem of this approximation is the low variance of the sample
    # drop from the g(x) distribution. Which we known from the example.
    # How we accelerate the approximation speed ?
    # Maybe condictional probability can do us some favors ?
    print 'the deflactor of first approximation: %.3f' % (
          CMonte_carlo().recovery()[0])
    print 'the deflactor of second approximation: %.3f' % (
          CMonte_carlo(25./38.).recovery()[0])

    print rare_event().estimate_p()

