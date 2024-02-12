from typing import Iterable, Literal
import autograd.numpy as np
from autograd import grad
import autograd.scipy.stats.norm as norm
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam, sgd

def gaussian(x, mu, sigma):
    return np.exp(-(((x-mu)/sigma)**2)/2)

class TS_FNN():
    def __init__(self, antecedents_sets_count: list[int], rules_antecedent_sets: list[list[int]], t_norm: Literal['min', 'prod'] = 'min'):
        ''' Creates Fuzzy Neural Network equivalent to Takagi-Sugeno rule base.
            Parameters:
            antecedents_sets_count - list of integers that defies the size of the term set of each antecedent
            rules_antecedent_sets - list of lists of integers that defies the index of the term of each antecedent of each rule
            t_norm - {min, prod} - defies which t-norm to be used for calculating the firing strength of each rule
        '''
        self.antecedents_sets_count = antecedents_sets_count
        self.rules_count = len(rules_antecedent_sets)
        self.antecedents_count = len(antecedents_sets_count)
        self.rules_antecedent_sets = rules_antecedent_sets

        self.zero_parameters()

        if t_norm == 'min':
            self.t_norm = lambda x: np.min(x)
        elif t_norm == 'prod':
            self.t_norm = lambda x: np.prod(x)

    def init_mu_parameters(self, mus: list[list[float]]):
        ''' Initializes the mu of the gaussian membership function of each term set of each antecedent
            Parameter:
            mus - list of list of floats
        '''
        self.parameters = (np.array(mus), self.parameters[1], self.parameters[2], self.parameters[3])

    def init_sigma_parameters(self, sigma):
        ''' Initializes the sigma of the gaussian membership function of each term set of each antecedent
            Parameter:
            sigma - list of list of floats
        '''
        self.parameters = (self.parameters[0], np.array(sigma), self.parameters[2], self.parameters[3])

    def zero_parameters(self):
        ''' Initializes all parameters to zeros
        '''
        self.parameters = ([np.zeros(sets_count) for sets_count in self.antecedents_sets_count],
                        [np.ones(sets_count) for sets_count in self.antecedents_sets_count],
                        np.ones(self.rules_count),
                        [np.ones(len(antecedent_count)) for antecedent_count in self.rules_antecedent_sets])

    def forward(self, parameters, x):
        ''' Calculates the forward pass in the FNN with the specified parameters and input x
            Parameters:
            parameters - the weights of the FNN
            x - the input
        '''
        antecedent_sets_mu, antecedent_sets_sigma, rules_bias, rules_parameters = parameters
        sum_firing_strengths = 0
        sum_rule_results = 0
        for rule_index, rule in enumerate(self.rules_antecedent_sets):
            firing_strength = self.t_norm(
                np.array([
                    gaussian(x[antecedent_index],
                    antecedent_sets_mu[antecedent_index][antecedent_set_index],
                    antecedent_sets_sigma[antecedent_index][antecedent_set_index])
                for antecedent_index, antecedent_set_index in enumerate(rule)])
            )
            sum_firing_strengths = sum_firing_strengths + firing_strength

            rule_result = firing_strength*(np.dot(x, rules_parameters[rule_index]) + rules_bias[rule_index])
            sum_rule_results = sum_rule_results + rule_result
        return sum_rule_results/sum_firing_strengths

    def predict(self, x):
        ''' Calculates the forward pass in the FNN on the input x
            Parameters:
            x - the input
        '''
        return self.forward(self.parameters, x)

    def fit(self, X, Y,
            gradient_descent_strategy: Literal['stochastic', 'full'] = 'stochastic',
            loss_f: Literal['squared_sum', 'log_likelihood'] = 'squared_sum',
            gradient_accumulation_step: int = 32,
            epochs: int = 10,
            learning_rate: float = 0.001,
            L2_reg: float = 0.1,
            noise_variance: float = 0.01,
            noise_scale: float = 0.1
        ):
        ''' Trains the FNN on the X and labels Y
            Parameters:
            X - training data
            Y - labels
            gradient_descent_strategy = {stochastic, full} - defies the strategy for gradient descent
            loss_f = {squared_sum, log_likelihood} - defies the used loss function
            gradient_accumulation_step - the number of samples for accumulating the gradient before updating the weights,
                similar to batch size, works only with stochastic
            epochs - the number of training epochs
            learning rate - the learning rate of the gradient descent
            L2 reg - the coefficient for l2 regularization when using squared sum loss
            noise_variance - the variance of the noise when using squared sum loss
            noise_scale - the scale of the noise when using log likelihood loss
        '''
        # training examples should be vectors with length equal to antecedents_count
        assert(X.shape[1] == self.antecedents_count)

        def logprob(parameters, X, Y):
            if loss_f == 'squared_sum':
                flatten_params, _ = flatten(parameters)
                log_prior = -L2_reg * np.sum(flatten_params**2)
                Y_bar = np.array([self.forward(parameters, x) for x in X])
                log_lik = -np.sum((Y_bar - Y)**2) / noise_variance
                return log_prior + log_lik
            elif loss_f == 'log_likelihood':
                Y_bar = np.array([self.forward(parameters, x) for x in X])
                return np.sum(norm.logpdf(Y_bar, Y, noise_scale))
            else: assert(False)

        idx = np.arange(len(X), dtype='int32')
        np.random.shuffle(idx)

        if gradient_descent_strategy == 'stochastic':
            i = 0
            # training loop
            for e in range(epochs):
                for b in range(0, len(idx), gradient_accumulation_step):
                    batch_X = [ X[i] for i in idx[b:min(b+gradient_accumulation_step, len(idx))] ]
                    batch_Y = [ Y[i] for i in idx[b:min(b+gradient_accumulation_step, len(idx))] ]
                    def loss(parameters, t):
                        return -logprob(parameters, batch_X, batch_Y)

                    def callback(params, t, g):
                        print("Iteration {} Epoch {} loss {}".format(i, e, loss(params, t)))

                    i += 1
                    self.parameters = sgd(grad(loss), self.parameters, step_size=learning_rate, num_iters=1, callback=callback)
        elif gradient_descent_strategy == 'full':
            X = [X[i] for i in idx]
            Y = [Y[i] for i in idx]
            def loss(parameters, t):
                    return -logprob(parameters, X, Y)

            def callback(params, t, g):
                print("Iteration {} loss {}".format(t, loss(params, t)))

            self.parameters = adam(grad(loss), self.parameters, step_size=learning_rate, num_iters=epoch, callback=callback)
