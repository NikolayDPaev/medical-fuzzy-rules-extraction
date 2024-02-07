import autograd.numpy as np
from autograd import grad
import autograd.scipy.stats.norm as norm
from autograd.misc.optimizers import adam, sgd

def gaussian(x, mu, sigma):
    return np.exp(-(((x-mu)/sigma)**2)/2)

class TS_FNN():
    def __init__(self, antecedents_sets_count: list[int], rules_antecedent_sets: list[list[int]], t_norm = 'min') -> None:
        self.antecedents_sets_count = antecedents_sets_count
        self.rules_count = len(rules_antecedent_sets)
        self.antecedents_count = len(antecedents_sets_count)
        self.rules_antecedent_sets = rules_antecedent_sets

        self.init_zero_parameters()

        if t_norm == 'min':
            self.t_norm = lambda x: np.min(x)
        elif t_norm == 'prod':
            self.t_norm = lambda x: np.prod(x)

    def init_zero_parameters(self):
        self.parameters = ([np.zeros(sets_count) for sets_count in self.antecedents_sets_count],
                        [np.ones(sets_count) for sets_count in self.antecedents_sets_count],
                        np.ones(self.rules_count),
                        [np.ones(len(antecedent_count)) for antecedent_count in self.rules_antecedent_sets])

    def predict(self, x):
        def forward(parameters, x):
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
        return forward(self.parameters, x)

    def fit(self, X, Y, gradient_accumulation_step = 32, epoch = 10):
        # training examples should be vectors with length equal to antecedents_count
        assert(X.shape[1] == self.antecedents_count)

        def forward(parameters, x):
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

        def logprob(parameters, X, Y, noise_scale=0.1):
            Y_bar = np.array([forward(parameters, x) for x in X])
            return np.sum(norm.logpdf(Y_bar, Y, noise_scale))

        idx = np.arange(len(X), dtype='int32')
        np.random.shuffle(idx)
        i = 0
        # training loop
        for e in range(epoch):
            for b in range(0, len(idx), gradient_accumulation_step):
                batch_X = [ X[i] for i in idx[b:min(b+gradient_accumulation_step, len(idx))] ]
                batch_Y = [ Y[i] for i in idx[b:min(b+gradient_accumulation_step, len(idx))] ]
                def loss(parameters, t):
                    return -logprob(parameters, batch_X, batch_Y)

                def callback(params, t, g):
                    print("Iteration {} Epoch {} log likelihood {}".format(i, e, -loss(params, t)))

                i += 1
                self.parameters = sgd(grad(loss), self.parameters, step_size=0.001, num_iters=1, callback=callback)

        # def loss(parameters, t):
        #     return -logprob(parameters, X, Y)

        # def callback(params, t, g):
        #     print("Iteration {} log likelihood {}".format(t, -loss(params, t)))

        # self.parameters = adam(grad(loss), self.parameters, step_size=0.1, num_iters=10, callback=callback)