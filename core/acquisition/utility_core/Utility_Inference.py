
from scipy.stats import dirichlet
import numpy as np
from .utilities import composed_utility_functions
import time
class prior_sample_generator:
    def __init__(self, u_funcs):
        self.u_funcs = u_funcs
        self.wdim = len(u_funcs)
        self.tindvdim = [u.n_params for u in u_funcs]
        self.tdim = np.sum([u.n_params for u in u_funcs])

    def __call__(self, n_samples=1, seed=None):

        if seed is None:
            theta_samples = [self.dirich_sampler(dim=d, n_samples=n_samples) for d in self.tindvdim]
            weight_samples = self.dirich_sampler(self.wdim, n_samples=n_samples)
        else:

            theta_samples = [self.dirich_sampler(d, n_samples=n_samples, seed=seed) for d in self.tindvdim]
            weight_samples = self.dirich_sampler(self.wdim, n_samples=n_samples, seed=seed)

        return theta_samples, weight_samples

    def dirich_sampler(self, dim, n_samples=1, seed=None):

        if seed is None:
            samples = np.random.dirichlet(np.ones((dim,)), n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.dirichlet(np.ones((dim,)), n_samples)
        return samples


import matplotlib.pyplot as plt
class Inference_method():
    def __init__(self, u_funcs, Likelihood_name="HardBounds"):
        self.u_funcs = u_funcs
        self.wdim = len(u_funcs)
        self.tindvdim = [u.n_params for u in u_funcs]
        self.tdim = np.sum([u.n_params for u in u_funcs])
        self.m_dim = self.tdim + self.wdim
        self.u_function = composed_utility_functions(u_funcs)
        self.posterior_samples = None
        self.Pareto_front  = None
        self.preferred_points = None
        self.Likelihood_name = Likelihood_name

    def update_sampled_data(self, Pareto_front, preferred_points):
        self.Pareto_front = Pareto_front
        self.preferred_points = preferred_points


        plt.scatter(Pareto_front[0][:,0], Pareto_front[0][:,1])
        plt.scatter(Pareto_front[0][preferred_points,0], Pareto_front[0][preferred_points,1], color="red")
        plt.show()
    def get_Decision_Maker_Data(self):
        return self.Pareto_front, self.preferred_points

    def get_utility_function(self):
        return self.u_function

    def prior(self, theta, weights):

        t_arr = np.hstack(theta)
        w_arr = weights
        param_arr = np.hstack([t_arr, w_arr])

        assert param_arr.shape[1] == self.m_dim, "wrong dimension"
        output_w = np.zeros((w_arr.shape[0], 1))

        for i in range(w_arr.shape[0]):
            try:
                output_w[i] = dirichlet.pdf(x=w_arr[i], alpha=np.ones((self.wdim,)))
            except:
                output_w[i] = 0

        output_t = np.zeros((w_arr.shape[0], self.wdim))
        for i in range(len(self.tindvdim)):
            for j in range(len(theta[i])):
                try:
                    output_t[j, i] = dirichlet.pdf(x=theta[i][j], alpha=np.ones((self.tindvdim[i],)))
                except:
                    output_t[j, i] = 0

        concat_vals = np.hstack([output_w, output_t])
        prior_fval = np.product(concat_vals, axis=1)
        return prior_fval

    def prior_sampler(self, n_samples, seed=None):
        if seed is None:
            theta_samples = [self.dirich_sampler(dim=d, n_samples=n_samples) for d in self.tindvdim]
            weight_samples = self.dirich_sampler(self.wdim, n_samples=n_samples)
        else:

            theta_samples = [self.dirich_sampler(d, n_samples=n_samples, seed=seed) for d in self.tindvdim]
            weight_samples = self.dirich_sampler(self.wdim, n_samples=n_samples, seed=seed)

        return theta_samples, weight_samples

    def dirich_sampler(self, dim, n_samples=1, seed=None):

        if seed is None:
            samples = np.random.dirichlet(np.ones((dim,)), n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.dirichlet(np.ones((dim,)), n_samples)
        return samples

    def posterior_sampler(self, n_samples, seed=None, warmup=100):
        # Metropolis-Hasting algorithm. proposal distribution is the dirichlet prior.
        np.random.seed(seed)
        accepted_samples = 0

        seed_prior_sample = self.prior_sampler(n_samples=1, seed=None)

        if self.Pareto_front is None:
            return self.prior_sampler(n_samples=n_samples, seed=None)

        lik0 = self.Likelihood(theta=seed_prior_sample[0],
                               weights=seed_prior_sample[1], log=False)

        samples_t = [np.zeros((n_samples + warmup, d)) for d in self.tindvdim]
        samples_w = np.zeros((n_samples + warmup, self.wdim))

        while accepted_samples < n_samples + warmup:
            candidate_sample = self.prior_sampler(n_samples=1, seed=None)

            lik_cadidate = self.Likelihood(theta=candidate_sample[0],
                                           weights=candidate_sample[1], log=False)

            ratio = lik_cadidate / lik0

            acceptance_prob = np.min([1, ratio])

            if acceptance_prob > np.random.random():
                lik0 = lik_cadidate

                for k in range(len(self.tindvdim)):
                    samples_t[k][accepted_samples] = candidate_sample[0][k]

                samples_w[accepted_samples] = candidate_sample[1]
                accepted_samples += 1

        removed_samples_t = []
        for st in samples_t:
            removed_samples_t.append(st[warmup:])

        self.posterior_samples = removed_samples_t, samples_w[warmup:]
        return removed_samples_t, samples_w[warmup:]

    def get_generated_posterior_samples(self):
        return self.posterior_samples

    def Likelihood(self, theta, weights, log=False, verbose=False):

        if self.Likelihood_name == "HardBounds":
            return self.HardBoundsModel( theta, weights)#self.PlacketLuceModel(self, theta, weights)
        elif self.Likelihood_name == "PlacketLuce":
            return self.PlacketLuceModel(theta, weights)

    def HardBoundsModel(self, theta, weights, log=False, verbose=False):

        N = len(self.Pareto_front)
        weights = np.array(weights)
        log_lik = np.zeros((N, weights.shape[0]))

        for n in range(N):
            preferred_point = self.Pareto_front[n][self.preferred_points[n]]

            u_pf_samples = self.u_function(y=self.Pareto_front[n],
                                           weights=weights,
                                           parameters=theta)

            simulated_best_index = np.argmax(u_pf_samples)
            decision_maker_best_index = self.preferred_points[n]

            if simulated_best_index==decision_maker_best_index:
                likelihood = 1
            else:
                likelihood = 0

            log_lik[n, :] = likelihood

        Lik_val = np.product(log_lik, axis=0)

        return Lik_val

    def PlacketLuceModel(self, theta, weights, log=False, verbose=False):

        N = len(self.Pareto_front)
        weights = np.array(weights)
        log_lik = np.zeros((N, weights.shape[0]))

        for n in range(N):
            preferred_point = self.Pareto_front[n][self.preferred_points[n]]

            u_best = self.u_function(y=preferred_point,
                                     weights=weights,
                                     parameters=theta).reshape(-1)

            u_pf_samples = self.u_function(y=self.Pareto_front[n],
                                           weights=weights,
                                           parameters=theta)

            likelihood = np.exp(u_best) / np.sum(np.exp(u_pf_samples), axis=-1)

            log_lik[n, :] = likelihood

        Lik_val = np.product(log_lik, axis=0)

        return Lik_val

    def Expected_likelihood(self, preferred_point,
                            sampled_pareto_front,
                            posterior_samples=None):

        preferred_point = np.atleast_2d(preferred_point)
        if posterior_samples is None:
            utility_parameters, linear_weight_combination= self.posterior_sampler(n_samples=50,
                                                                        seed=None)
        else:
            utility_parameters, linear_weight_combination = posterior_samples

        utility_parameters = np.array(utility_parameters).squeeze()
        utility_parameters = np.atleast_2d(utility_parameters)

        N_parameter_samples = utility_parameters.shape[0]
        lik = np.zeros((N_parameter_samples))


        Pareto_front = np.concatenate((preferred_point,
                                       sampled_pareto_front))

        u_pf_samples = self.u_function(y=Pareto_front,
                                       weights=linear_weight_combination,
                                       parameters=utility_parameters)


        simulated_best_index = np.argmax(u_pf_samples, axis=1)
        # print(simulated_best_index)

        lik[simulated_best_index==0] = 1

        expected_likelihood = np.mean(lik)

        return expected_likelihood

    def Expected_Utility(self, preferred_point, posterior_samples=None):

        if posterior_samples is None:
            posterior_theta, posterior_weights = self.posterior_sampler(n_samples=50,
                                                                        seed=None)
        else:
            posterior_theta, posterior_weights = posterior_samples

        preferred_point = np.atleast_2d(preferred_point)
        predictive_lik = np.zeros((preferred_point.shape[0],))
        for l in range(len(preferred_point)):
            u_best = self.u_function(y=preferred_point[l],
                                     weights=posterior_weights,
                                     parameters=posterior_theta).reshape(-1)

            likelihood_v2 = u_best

            predictive_lik[l] = np.mean(likelihood_v2)

        return predictive_lik


