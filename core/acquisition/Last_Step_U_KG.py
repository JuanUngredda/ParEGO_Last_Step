# Copyright (c) 2018, Raul Astudillo Marban

import numpy as np
from GPyOpt.experiment_design import initial_design
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.experiment_design import initial_design
from aux_modules.gradient_modules import gradients
from scipy.stats import norm
import scipy
import time
import matplotlib.pyplot as plt
from pygmo import hypervolume


class AcquisitionUKG(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details. 
    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, model_c=None, alpha=1.95,optimizer=None, cost_withGradients=None, utility=None, true_func=None):
        self.optimizer = optimizer
        self.utility = utility
        self.MCMC = False
        self.n_samples = 3
        self.counter = 0
        self.current_max_value = np.inf
        self.Z_samples_obj = None
        self.Z_samples_const = None
        self.true_func = true_func
        self.saved_Nx = -10
        self.name = "Constrained_UKG"
        if model_c is None:
            self.constraints_flag = False
        else:
            self.constraints_flag = True
        print(alpha)
        print(cost_withGradients)
        print(model_c)
        super(AcquisitionUKG, self).__init__(model, space, optimizer, model_c = model_c, alpha =alpha, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, X):
        """
        Computes the aquisition function

        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        self.Z_samples_obj = np.array([0])#np.random.normal(size=n_z)#np.array([-2.326, -1.282, 0, 1.282, 2.326]) #np.array([-2.326, -1.282, 0, 1.282, 2.326])
        self.Z_samples_const = np.array([0]) #np.random.normal(size=n_z)
        full_support = False  # If true, the full support of the utility function distribution will be used when computing the acquisition function value.
        X = np.atleast_2d(X)
        utility_params_samples = self.utility.sample_parameter(self.n_samples)
        self.utility_params_samples = utility_params_samples
        if full_support:
            utility_params_samples = self.utility.parameter_dist.support
            utility_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        marginal_acqX = self._marginal_acq(X, utility_params_samples)
        if full_support:
            acqX = np.matmul(marginal_acqX, utility_params_samples)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(utility_params_samples)

        acqX = np.reshape(acqX, (X.shape[0], 1))
        return -acqX

    def _marginal_acq(self, X, utility_params_samples):
        """
        """
        marginal_acqX = np.zeros((X.shape[0], len(utility_params_samples)))
        n_h = 1  # Number of GP hyperparameters samples.
        # self.model.restart_hyperparameters_counter()
        # gp_hyperparameters_samples = self.model.get_hyperparameters_samples(n_h)

        n_z = len(self.Z_samples_obj)#2  # Number of samples of Z.
        Z_samples = self.Z_samples_obj #np.random.normal(size=n_z)
        Z_samples_c = self.Z_samples_const

        for h in range(n_h):
            # self.model.set_hyperparameters(h)
            varX_c = self.model_c.posterior_variance(X, noise=True)

            for i in range(0, len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)

                self.model_c.partial_precomputation_for_covariance(x)
                self.model_c.partial_precomputation_for_covariance_gradient(x)

                for l in range(0, len(utility_params_samples)):

                    aux_c = np.reciprocal(varX_c[:,i])

                    for z in range(len(Z_samples)):
                        # inner function of maKG acquisition function.
                        def inner_func(X_inner, plots=False, include_best=None, include_val=None):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner

                            func_val = np.reshape(a, (len(X_inner), 1))

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_samples_c[z], aux=aux_c,
                                               X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))
                            Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                            func_val_constrained = func_val * Fz

                            if plots:
                                print("func_val", func_val, "Fz", Fz)
                                fig, axs = plt.subplots(2, 2)
                                true_func_values = self.true_func.evaluate(X_inner)
                                best_scaled_val_idx = np.argmin(func_val_constrained)
                                predicted_best_mu = self.model.posterior_mean(include_best)
                                print("best_scaled_val",np.min(func_val_constrained), "optimised best", include_val)
                                print("utility_params_samples[l]",utility_params_samples[l])
                                feasable_mu_index = np.array(Fz > 0.51, dtype=bool).reshape(-1)
                                axs[0, 0].set_title("Inner_mu_plot")
                                axs[0, 0].scatter(muX_inner[0][feasable_mu_index],muX_inner [1][feasable_mu_index], c=np.array(a).reshape(-1)[feasable_mu_index])
                                axs[0, 0].scatter(muX_inner[0][best_scaled_val_idx], muX_inner[1][best_scaled_val_idx], color="red")
                                axs[0, 0].scatter(predicted_best_mu[0], predicted_best_mu[1], color="black")
                                axs[0, 0].legend()

                                axs[0, 1].set_title("predicted vs real f1")
                                axs[0, 1].scatter(muX_inner[0], np.array(true_func_values[0][0]).reshape(-1))
                                axs[0, 1].legend()

                                axs[1, 0].set_title("predicted vs real f2")
                                axs[1, 0].scatter(muX_inner[1], np.array(true_func_values[0][1]).reshape(-1))
                                axs[1, 0].legend()
                                plt.show()

                            return func_val_constrained

                        # inner function of maKG acquisition function with its gradient.
                        def inner_func_with_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)

                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            da_dX_inner = np.tensordot(utility_params_samples[l], dmu_dX_inner, axes=1)

                            func_val = np.reshape(a , (len(X_inner), 1))
                            func_gradient = np.reshape(da_dX_inner, X_inner.shape)

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_samples_c[z], aux=aux_c, X_inner=X_inner,
                                               precompute_grad=True)
                            Fz, grad_Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0,
                                                                                          gradient_flag=True)

                            func_val_constrained = func_val*Fz
                            func_gradient_constrained = np.array(func_val).reshape(-1) * grad_Fz.reshape(-1) + Fz.reshape(-1) * func_gradient.reshape(-1)

                            return func_val_constrained, func_gradient_constrained

                        # values = self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)
                        # X = initial_design('random', self.space, 10000)
                        # inner_func(X, plots=True, include_best=values[0], include_val = values[1])

                        x = np.atleast_2d(x)
                        marginal_acqX[i, l] += inner_func(x) #self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)[1]

        marginal_acqX = marginal_acqX / (n_z * n_h)
        return marginal_acqX

    def _compute_acq_withGradients(self, X):
        """
        """

        full_support = False  # If true, the full support of the utility function distribution will be used when computing the acquisition function value.
        X = np.atleast_2d(X)

        utility_params_samples = self.utility_params_samples
        if full_support:
            utility_params_samples = self.utility.parameter_dist.support
            utility_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples,
        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X, utility_params_samples)
        if full_support:
            acqX = np.matmul(marginal_acqX, utility_dist)
            dacq_dX = np.tensordot(marginal_dacq_dX, utility_dist, 1)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(utility_params_samples)
            dacq_dX = np.sum(marginal_dacq_dX , axis=2)/len(utility_params_samples)

        acqX = np.reshape(acqX, (X.shape[0], 1))
        dacq_dX = np.reshape(dacq_dX, X.shape)
        return acqX, dacq_dX

    def _marginal_acq_with_gradient(self, X, utility_params_samples):
        """
        """
        marginal_acqX = np.zeros((X.shape[0], len(utility_params_samples)))
        marginal_dacq_dX = np.zeros((X.shape[0], X.shape[1], len(utility_params_samples)))
        n_h = 1  # Number of GP hyperparameters samples.
        # gp_hyperparameters_samples = self.model.get_hyperparameters_samples(n_h)

        n_z = len(self.Z_samples_obj)#2  # Number of samples of Z.
        Z_samples = np.array([0]) # self.Z_samples_obj #np.random.normal(size=n_z)
        Z_samples_c = np.array([0]) #self.Z_samples_const

        for h in range(n_h):
            # self.model.set_hyperparameters(h)

            varX_c = self.model_c.posterior_variance(X, noise=True)
            for i in range(0, len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)

                self.model_c.partial_precomputation_for_covariance(x)
                self.model_c.partial_precomputation_for_covariance_gradient(x)

                for l in range(0, len(utility_params_samples)):
                    # Precompute aux1 and aux2 for computational efficiency.

                    aux_c = np.reciprocal(varX_c[:, i])
                    for z in range(len(Z_samples)):
                        # inner function of maKG acquisition function.
                        def inner_func(X_inner, plots=False, include_best=None, include_val=None):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner

                            func_val = np.reshape(a, (len(X_inner), 1))

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_samples_c[z], aux=aux_c,
                                               X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))
                            Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                            func_val_constrained = func_val * Fz

                            if plots:
                                print("func_val", func_val, "Fz", Fz)
                                fig, axs = plt.subplots(2, 2)
                                true_func_values = self.true_func.evaluate(X_inner)
                                best_scaled_val_idx = np.argmin(func_val_constrained)
                                predicted_best_mu = self.model.posterior_mean(include_best)
                                print("best_scaled_val", np.min(func_val_constrained), "optimised best", include_val)
                                print("utility_params_samples[l]", utility_params_samples[l])
                                feasable_mu_index = np.array(Fz > 0.51, dtype=bool).reshape(-1)
                                axs[0, 0].set_title("Inner_mu_plot")
                                axs[0, 0].scatter(muX_inner[0][feasable_mu_index], muX_inner[1][feasable_mu_index],
                                                  c=np.array(a).reshape(-1)[feasable_mu_index])
                                axs[0, 0].scatter(muX_inner[0][best_scaled_val_idx], muX_inner[1][best_scaled_val_idx],
                                                  color="red")
                                axs[0, 0].scatter(predicted_best_mu[0], predicted_best_mu[1], color="black")
                                axs[0, 0].legend()

                                axs[0, 1].set_title("predicted vs real f1")
                                axs[0, 1].scatter(muX_inner[0], np.array(true_func_values[0][0]).reshape(-1))
                                axs[0, 1].legend()

                                axs[1, 0].set_title("predicted vs real f2")
                                axs[1, 0].scatter(muX_inner[1], np.array(true_func_values[0][1]).reshape(-1))
                                axs[1, 0].legend()
                                plt.show()

                            return func_val_constrained

                        # inner function of maKG acquisition function with its gradient.
                        def inner_func_with_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)

                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            da_dX_inner = np.tensordot(utility_params_samples[l], dmu_dX_inner, axes=1)

                            func_val = np.reshape(a, (len(X_inner), 1))
                            func_gradient = np.reshape(da_dX_inner, X_inner.shape)

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_samples_c[z], aux=aux_c,
                                               X_inner=X_inner,
                                               precompute_grad=True)
                            Fz, grad_Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0,
                                                                                          gradient_flag=True)

                            func_val_constrained = func_val * Fz
                            func_gradient_constrained = np.array(func_val).reshape(-1) * grad_Fz.reshape(
                                -1) + Fz.reshape(-1) * func_gradient.reshape(-1)

                            return func_val_constrained, func_gradient_constrained

                        # x_opt, opt_val = self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)
                        x = np.atleast_2d(x)
                        opt_val, opt_val_grad = inner_func_with_gradient(x) #self.inner_func_with_gradient(x_opt)
                        marginal_acqX[i, l] += opt_val
                        marginal_dacq_dX[i, :, l] += opt_val_grad  #grad_f_val

        marginal_acqX = marginal_acqX / (n_h * n_z)
        marginal_dacq_dX = marginal_dacq_dX / (n_h * n_z)
        return -marginal_acqX, -marginal_dacq_dX

    def probability_feasibility_multi_gp_wrapper(self, model, l=0):
        def probability_feasibility_multi_gp(x):
            # print("model",model.output)
            x = np.atleast_2d(x)
            Fz = []
            for m in range(model.output_dim):
                Fz.append(self.probability_feasibility(x, model.output[m], l=0))
            Fz = np.product(Fz, axis=0)
            return Fz
        return probability_feasibility_multi_gp

    def probability_feasibility(self, x, model, l=0):

        mean = model.posterior_mean(x)
        var = model.posterior_variance(x, noise=False)
        std = np.sqrt(var).reshape(-1, 1)

        mean = mean.reshape(-1, 1)

        norm_dist = norm(mean, std)
        Fz = norm_dist.cdf(l)

        return Fz.reshape(-1, 1)


