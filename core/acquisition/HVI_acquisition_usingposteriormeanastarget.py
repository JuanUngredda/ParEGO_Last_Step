# Copyright (c) 2018, Raul Astudillo Marban

import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.experiment_design import initial_design
from aux_modules.gradient_modules import gradients
from scipy.stats import norm
import scipy
import time
import matplotlib.pyplot as plt
from pygmo import hypervolume
from pygmo import *
from pyDOE import *

class HVI(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details. 
    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, alpha=1.95,optimizer=None,ref_point=None,cost_withGradients=None, Inference_Object=None):

        self.MCMC = False
        self.output_dimensions = model.output_dim
        self.counter = 0
        self.Inference_Object = Inference_Object
        self.name = "Constrained_HVI"
        self.ref_point = ref_point
        self.alpha = alpha
        self.n_samples=50
        self.old_number_of_simulation_samples = 0
        self.old_number_of_dm_samples = 0
        self.true_utility_values=None
        self.historic_X = 0
        self.posterior_samples = self.get_posterior_samples()
        super(HVI, self).__init__(model, space, optimizer,alpha=alpha, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients


    def include_fantasised_posterior_samples(self, posterior_samples):
        self.fantasised_posterior_samples = posterior_samples

    def get_posterior_samples(self):
        # posterior_samples = self.Inference_Object.get_generated_posterior_samples()
        # if posterior_samples is None:
        posterior_samples = self.Inference_Object.posterior_sampler(self.n_samples)
        return posterior_samples

    def get_posterior_utility_landscape(self, y):

        if self.true_utility_values is None:
            posterior_utility_samples = self.Inference_Object.Expected_Utility(y,
                                                                               posterior_samples=self.posterior_samples)
        else:
            posterior_utility_samples = self.true_utility_values(y)
        return posterior_utility_samples

    def include_true_dm_utility_vals(self, utility):
        self.true_utility_values = utility


    def _compute_acq(self, X):
        """
        Computes the aquisition function
        
        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        # print("_compute_acq")

        X =np.atleast_2d(X)
        if self.ref_point is None:
            self.ref_point = self.ref_point_computation()
            self.alpha = 1.95


        current_number_simulation_points = self.model.get_X_values().shape[0]

        Pareto_front, preferred_points = self.Inference_Object.get_Decision_Maker_Data()
        if preferred_points is None:
            current_number_dm_points = 0
        else:
            current_number_dm_points = len(preferred_points)
        if not self.old_number_of_simulation_samples == current_number_simulation_points:
            self.old_number_of_simulation_samples = current_number_simulation_points

            print("current dm points", current_number_dm_points)
            if not self.old_number_of_dm_samples== current_number_dm_points:
                self.posterior_samples = self.get_posterior_samples()
            self._update_current_extremum()

        HVI = self.compute_HVI_LCB(X).reshape(-1)

        return HVI

    def Generate_Pareto_Front(self):
        X_train = self.model.get_X_values()
        GP_y_predictions  = self.mean_prediction_model(X_train ,self.model)


        bounds = self.space.get_continuous_bounds()
        bounds = self.bounds_format_adapter(bounds)
        udp = GA(f=GP_y_predictions, bounds=bounds, n_obj=self.output_dimensions)
        pop = population(prob=udp, size=40)
        algo = algorithm(nsga2(gen=300))
        pop = algo.evolve(pop)
        fits, vectors = pop.get_f(), pop.get_x()
        ndf, dl, dc, ndr = fast_non_dominated_sorting(fits)
        result_x = vectors[ndf[0]]
        result_fx = fits[ndf[0]]

        return -result_fx  # GP_y_predictions

    def bounds_format_adapter(self, bounds):
        bounds = np.array(bounds)
        bounds_correct_format = []
        for b in range(bounds.shape[0]):
            bounds_correct_format.append(list(bounds[:, b]))
        return bounds_correct_format

    def mean_prediction_model(self, X, model):
        def prediction(X):
            X = np.atleast_2d(X)

            mu_x = model.posterior_mean(X)
            mu_x = np.vstack(mu_x).T
            return mu_x
        return prediction

    def compute_HVI_LCB(self, X):
        X = np.atleast_2d(X)

        if X.shape[0] != self.historic_X:
            self.historic_X = X.shape[0]
            self.non_dominated_surface = self.Generate_Pareto_Front()

        non_dominated_surface = self.non_dominated_surface
        P = (- non_dominated_surface).tolist()
        P_cur =  (-non_dominated_surface).tolist()


        HV0_func = hypervolume(P_cur)
        HV0 = HV0_func.compute(ref_point=self.ref_point)
        HVvals = np.zeros(X.shape[0])
        Expected_utility_vals = np.zeros(X.shape[0])
        for i in range(len(X)):
            x = np.atleast_2d(X[i])
            ## Hypervolume computations
            mu = -self.model.posterior_mean(x)
            var = self.model.posterior_variance(x, noise=False)

            y_lcb = mu - self.alpha * np.sqrt(var)
            y_lcb = np.transpose(y_lcb)
            P_new = P
            P_new = np.vstack([P_new, y_lcb])
            hv_new = hypervolume(P_new.tolist())

            try:
                HV_new= hv_new.compute(ref_point = self.ref_point)
            except:
                print("warwning! points outside reference")
                HV_new =0
            HVvals[i] = HV_new - HV0

        HVvals[HVvals < 0.0] = 0.0

        Normalised_Expected_Utility = (Expected_utility_vals- self.min_util)/(self.max_util- self.min_util)

        return  HVvals #* Normalised_Expected_Utility

    def ref_point_computation(self):
        ref_point_vector = []
        for m in range(self.model.output_dim):
            dim_m_conf_bound = self.model_mean_var_wrapper(j = m, alpha_ref_point=0.0)
            inner_opt_x, inner_opt_val = self.optimizer.optimize(f=dim_m_conf_bound)
            ref_point_vector.append(-float(inner_opt_val))
        ref_point_vector = np.array(ref_point_vector) + 0.1

        return list(ref_point_vector)

    def model_mean_var_wrapper(self, j, alpha_ref_point=None):
        if alpha_ref_point is not None:
            alpha =alpha_ref_point
        else:
            alpha = self.alpha
        def model_mean_var(X):
            X = np.atleast_2d(X)
            mu = -self.model.posterior_mean_builder(X, j)
            var = self.model.posterior_var_builder(X, j)
            y_lcb = mu - alpha * np.sqrt(var)

            return  -np.array(y_lcb).reshape(-1)
        return model_mean_var


    def _update_current_extremum(self):

        def current_max(X_inner):
            mu = self.model.posterior_mean(X_inner)[0]
            mu = mu.reshape(-1, 1)
            utility = []
            for m in mu:
                utility.append(self.get_posterior_utility_landscape(m.reshape(-1)).reshape(-1))
            return -np.array(utility).reshape(-1)

        def current_min(X_inner):
            mu = self.model.posterior_mean(X_inner)[0]
            mu = mu.reshape(-1, 1)
            utility = []
            for m in mu:
                utility.append(self.get_posterior_utility_landscape(m.reshape(-1)).reshape(-1))
            return np.array(utility).reshape(-1)

        inner_opt_x, self.max_util = self.optimizer.optimize(f=current_max, f_df=None, num_samples=100,
                                                             verbose=False)

        inner_opt_x, self.min_util = self.optimizer.optimize(f=current_min, f_df=None, num_samples=100,
                                                             verbose=False)


        # print("inner_opt_x, inner_opt_val",inner_opt_x, inner_opt_val)

class GA:
    # Define objectives
    def __init__(self, f, bounds, n_obj):
        self.f = f
        self.bounds = bounds
        self.n_obj = n_obj
    def fitness(self, x):
        x = np.atleast_2d(x)
        output = self.f(x)
        vect_func_val = []
        for i in range(self.n_obj):
            vect_func_val.append(output[:,i])
        return -np.array(vect_func_val).reshape(-1)

    # Return number of objectives
    def get_nobj(self):
        return self.n_obj

    # Return bounds of decision variables
    def get_bounds(self):
        return self.bounds  # ([0]*1, [2]*1)

    # Return function name
    def get_name(self):
        return "INNER OPTIMISATION PROBLEM"