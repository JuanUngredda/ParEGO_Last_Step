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
from pyDOE import *
from scipy.stats.distributions import norm

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
        self.n_samples = 5
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
        n_z = 3
        lhs_utility_samples  = lhs(self.model.output_dim, samples=self.n_samples)
        lhs_objective_samples = lhs(1, samples=n_z)
        lhs_constraint_samples = lhs(1, samples=n_z)
        self.utility_params_samples  = np.multiply(lhs_utility_samples, np.reciprocal(np.sum(lhs_utility_samples , axis=1)).reshape(-1,1))
        #utility_params_samples = self.utility.sample_parameter(self.n_samples)
        #self.utility_params_samples = utility_params_samples

        self.Z_samples_obj = norm(loc=0, scale=1).ppf(lhs_objective_samples) #np.array([-2.326, -1.282, 0, 1.282, 2.326])
        self.Z_samples_const = norm(loc=0, scale=1).ppf(lhs_constraint_samples )#np.random.normal(size=n_z)

        full_support = False  # If true, the full support of the utility function distribution will be used when computing the acquisition function value.
        X = np.atleast_2d(X)
        if full_support:
            utility_params_samples = self.utility.parameter_dist.support
            utility_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        marginal_acqX = self._marginal_acq(X, self.utility_params_samples)
        if full_support:
            acqX = np.matmul(marginal_acqX, self.utility_params_samples)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(self.utility_params_samples)

        acqX = np.reshape(acqX, (X.shape[0], 1))
        return acqX

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
            varX = self.model.posterior_variance(X, noise=True)
            varX_c = self.model_c.posterior_variance(X, noise=True)

            for i in range(0, len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)

                self.model_c.partial_precomputation_for_covariance(x)
                self.model_c.partial_precomputation_for_covariance_gradient(x)

                for l in range(0, len(utility_params_samples)):
                    aux = np.multiply(np.square(utility_params_samples[l]), np.reciprocal(
                        varX[:, i]))  # Precompute this quantity for computational efficiency.

                    aux_c = np.reciprocal(varX_c[:,i])

                    for z in range(len(Z_samples)):
                        # inner function of maKG acquisition function.
                        def inner_func(X_inner, plots=False, include_best=None, include_val=None):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,0]
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            b = np.sqrt(np.matmul(aux, np.square(cov)))

                            func_val = np.reshape(a + b * Z_samples[z], (len(X_inner), 1))

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_samples_c[z], aux=aux_c,
                                               X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))
                            Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                            func_val_constrained = func_val * Fz

                            if plots:
                                print("func_val", func_val, "Fz", Fz)
                                fig, axs = plt.subplots(2, 2)
                                true_func_values = self.true_func.evaluate(X_inner)
                                best_scaled_val_idx = np.argmax(func_val_constrained)
                                predicted_best_mu = self.model.posterior_mean(include_best)
                                print("best_scaled_val",np.min(func_val_constrained), "optimised best", include_val)
                                feasable_mu_index = np.array(Fz > 0.90, dtype=bool).reshape(-1)
                                axs[0, 0].set_title("Inner_mu_plots " + str(utility_params_samples[l]))
                                axs[0, 0].scatter(muX_inner[0][feasable_mu_index],muX_inner [1][feasable_mu_index], c=np.array(a).reshape(-1)[feasable_mu_index])
                                # axs[0, 0].scatter(muX_inner[0][best_scaled_val_idx], muX_inner[1][best_scaled_val_idx], color="red")
                                axs[0, 0].scatter(predicted_best_mu[0], predicted_best_mu[1], color="black")
                                axs[0,0].scatter(x[:,0], x[:,1], color="red")
                                axs[0, 0].legend()

                                axs[0, 1].set_title("predicted vs real f1")
                                axs[0, 1].scatter(muX_inner[0], np.array(true_func_values[0][0]).reshape(-1))
                                axs[0, 1].legend()

                                axs[1, 0].set_title("predicted vs real f2")
                                axs[1, 0].scatter(muX_inner[1], np.array(true_func_values[0][1]).reshape(-1))
                                axs[1, 0].legend()
                                plt.show()

                            return -func_val_constrained

                        # inner function of maKG acquisition function with its gradient.
                        def inner_func_with_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner, x)
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            da_dX_inner = np.tensordot(utility_params_samples[l], dmu_dX_inner, axes=1)
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            for k in range(X_inner.shape[1]):
                                dcov_dX_inner[:, :, k] = np.multiply(cov, dcov_dX_inner[:, :, k])
                            db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                            func_val = np.reshape(a + b * Z_samples[z], (len(X_inner), 1))
                            func_gradient = np.reshape(da_dX_inner + db_dX_inner * Z_samples[z], X_inner.shape)

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_samples_c[z], aux=aux_c, X_inner=X_inner,
                                               precompute_grad=True)
                            Fz, grad_Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0,
                                                                                          gradient_flag=True)

                            func_val_constrained = func_val*Fz
                            func_gradient_constrained = np.array(func_val).reshape(-1) * grad_Fz.reshape(-1) + Fz.reshape(-1) * func_gradient.reshape(-1)

                            return -func_val_constrained, -func_gradient_constrained

                        # values = self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)
                        # X = initial_design('random', self.space, 10000)
                        # inner_func(X, plots=True, include_best=values[0], include_val = values[1])


                        marginal_acqX[i, l] -=self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)[1]


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
            acqX = np.matmul(marginal_acqX, utility_params_samples)
            dacq_dX = np.tensordot(marginal_dacq_dX, utility_params_samples, 1)
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
        Z_samples = self.Z_samples_obj #np.random.normal(size=n_z)
        Z_samples_c = self.Z_samples_const

        for h in range(n_h):
            # self.model.set_hyperparameters(h)
            varX = self.model.posterior_variance(X, noise=True)
            dvar_dX = self.model.posterior_variance_gradient(X)

            varX_c = self.model_c.posterior_variance(X, noise=True)
            dvar_c_dX = self.model_c.posterior_variance_gradient(X)
            for i in range(0, len(X)):
                x = np.atleast_2d(X[i])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)

                self.model_c.partial_precomputation_for_covariance(x)
                self.model_c.partial_precomputation_for_covariance_gradient(x)

                for l in range(0, len(utility_params_samples)):
                    # Precompute aux1 and aux2 for computational efficiency.
                    aux = np.multiply(np.square(utility_params_samples[l]), np.reciprocal(varX[:, i]))
                    aux2 = np.multiply(np.square(utility_params_samples[l]), np.square(np.reciprocal(varX[:, i])))

                    aux_c = np.reciprocal(varX_c[:, i])
                    aux2_c = np.square(np.reciprocal(varX_c[:, i]))
                    for z in range(len(Z_samples)):
                        # inner function of maKG acquisition function.
                        def inner_func(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            func_val = np.reshape(a + b * Z_samples[z], (len(X_inner), 1))

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_samples_c[z], aux=aux_c,
                                               X_inner=X_inner)  # , test_samples = initial_design('random', self.space, 1000))
                            Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0)

                            func_val_constrained = func_val*Fz
                            return -func_val_constrained

                        # inner function of maKG acquisition function with its gradient.
                        def inner_func_with_gradient(X_inner):
                            X_inner = np.atleast_2d(X_inner)
                            muX_inner = self.model.posterior_mean(X_inner)
                            dmu_dX_inner = self.model.posterior_mean_gradient(X_inner)
                            cov = self.model.posterior_covariance_between_points_partially_precomputed(X_inner, x)[:, :,
                                  0]
                            dcov_dX_inner = self.model.posterior_covariance_gradient_partially_precomputed(X_inner, x)
                            a = np.matmul(utility_params_samples[l], muX_inner)
                            # a = support[t]*muX_inner
                            da_dX_inner = np.tensordot(utility_params_samples[l], dmu_dX_inner, axes=1)
                            b = np.sqrt(np.matmul(aux, np.square(cov)))
                            for k in range(X_inner.shape[1]):
                                dcov_dX_inner[:, :, k] = np.multiply(cov, dcov_dX_inner[:, :, k])
                            db_dX_inner = np.tensordot(aux, dcov_dX_inner, axes=1)
                            db_dX_inner = np.multiply(np.reciprocal(b), db_dX_inner.T).T
                            func_val = np.reshape(a + b * Z_samples[z], (len(X_inner), 1))
                            func_gradient = np.reshape(da_dX_inner + db_dX_inner * Z_samples[z], X_inner.shape)

                            grad_c = gradients(x_new=x, model=self.model_c, Z=Z_samples_c[z], aux=aux_c, X_inner=X_inner,
                                               precompute_grad=True)
                            Fz, grad_Fz = grad_c.compute_probability_feasibility_multi_gp(x=X_inner, l=0,
                                                                                          gradient_flag=True)

                            func_val_constrained = func_val*Fz
                            func_gradient_constrained = np.array(func_val).reshape(-1) * grad_Fz.reshape(-1) + Fz.reshape(-1) * func_gradient.reshape(-1)

                            return -func_val_constrained, -func_gradient_constrained

                        x_opt, opt_val = self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)
                        marginal_acqX[i, l] -= opt_val
                        x_opt = np.atleast_2d(x_opt)

                        #mu x opt calculations
                        muX_inner = self.model.posterior_mean(x_opt)
                        cov = self.model.posterior_covariance_between_points_partially_precomputed(x_opt, x)[:, :,
                              0]
                        a = np.matmul(utility_params_samples[l], muX_inner)
                        b = np.sqrt(np.matmul(aux, np.square(cov)))
                        mu_xopt = np.reshape(a + b * Z_samples[z], (len(x_opt), 1))

                        #grad x opt calculations
                        cov_opt = self.model.posterior_covariance_between_points_partially_precomputed(x_opt, x)[:, 0, 0]
                        dcov_opt_dx = self.model.posterior_covariance_gradient(x, x_opt)[:, 0, :]
                        b = np.sqrt(np.dot(aux, np.square(cov_opt)))
                        grad_mu_xopt = 0.5 * Z_samples[z] * np.reciprocal(b) * np.matmul(aux2, (2 * np.multiply(varX[:, i] * cov_opt, dcov_opt_dx.T) - np.multiply(np.square(cov_opt), dvar_dX[:, i, :].T)).T)

                        grad_c = gradients(x_new=x, model=self.model_c, Z=Z_samples_c[z], xopt=x_opt, aux=aux_c, aux2=aux2_c,
                                           varX=varX_c[:, i], dvar_dX=dvar_c_dX[:, i, :],
                                           test_samples=initial_design('random', self.space, 1000))

                        Fz_xopt, grad_Fz_xopt = grad_c.compute_probability_feasibility_multi_gp_xopt(xopt=x_opt,
                                                                                                     gradient_flag=True)

                        grad_f_val_xopt = np.array(mu_xopt).reshape(-1) * np.array(grad_Fz_xopt).reshape(-1) + np.array(
                            Fz_xopt).reshape(-1) * np.array(grad_mu_xopt).reshape(-1)


                        marginal_dacq_dX[i, :, l] = grad_f_val_xopt # grad_f_val

        marginal_acqX = marginal_acqX / (n_h * n_z)
        marginal_dacq_dX = marginal_dacq_dX / (n_h * n_z)
        return marginal_acqX, marginal_dacq_dX

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

