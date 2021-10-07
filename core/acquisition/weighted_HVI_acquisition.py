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
from pygmo import (hypervolume,
                    fast_non_dominated_sorting,
                   pareto_dominance)

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

            posterior_reference_utility = self.Inference_Object.Expected_Utility(-np.array(self.ref_point),
                                                                                 posterior_samples=self.posterior_samples)

            posterior_surface = posterior_utility_samples - posterior_reference_utility


            # Y = self.model.get_Y_values()
            # Y = np.concatenate(Y, axis=1)
            #
            #
            #
            # posterior_utility_samples = self.Inference_Object.Expected_likelihood(y,
            #                                                                       sampled_pareto_front=Y,
            #                                                                       posterior_samples=self.posterior_samples)

        else:


            posterior_utility_samples = self.Inference_Object.Expected_Utility(y,
                                                                              posterior_samples=self.true_utility_parameters)

            posterior_reference_utility = self.Inference_Object.Expected_Utility(-np.array(self.ref_point),
                                                                              posterior_samples=self.true_utility_parameters)

            posterior_surface = posterior_utility_samples - posterior_reference_utility
            # posterior_utility_samples = self.Inference_Object.Expected_likelihood(y,
            #                                                                       sampled_pareto_front=Y,
            #                                                                       posterior_samples=self.true_utility_parameters)

        return posterior_surface #posterior_utility_samples

    def include_true_dm_utility_vals(self, utility):
        self.true_utility_values = utility

    def include_true_dm_utility_parameters(self, parameters):
        self.true_utility_parameters = parameters

    def _compute_acq(self, X, verbose=False):
        """
        Computes the aquisition function
        
        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        # print("_compute_acq")
        self.verbose=verbose

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
            self.generated_mu_values = -self.generate_mu_values()
            # self.PF, self.PF_Xvals = self.Generate_Pareto_Front()  # generate pareto front
            print("current dm points", current_number_dm_points)

            if True:
                mu_vals = self.generate_mu_values()
                print(mu_vals)
                uvals = []
                for m in mu_vals:
                    uvals.append(self.get_posterior_utility_landscape(m))

                # plt.scatter(mu_vals[:,0], mu_vals[:,1], c=np.array(uvals).reshape(-1))
                # plt.show()

            if not self.old_number_of_dm_samples== current_number_dm_points:
                self.posterior_samples = self.get_posterior_samples()


        HVI = self.compute_HVI_LCB(X).reshape(-1)

        return HVI


    def initial_filtering_hypervolume(self,
                                     sampled_pareto_solutions):

        #computes non dominated points from previusly sampled points
        ndf, dl, dc, ndr = fast_non_dominated_sorting(points=sampled_pareto_solutions)
        non_dominated_indeces = ndf[0]
        non_dominated_front = np.array(sampled_pareto_solutions)[non_dominated_indeces] #non dominated sampled front

        generated_mu_values = self.generated_mu_values.copy()
        # generated_mu_values = np.concatenate((generated_mu_values, -self.PF))

        #
        # raise

        dominated_boolean_vector = []
        utility_vector = []
        non_dominated_vectors = []
        for muval in generated_mu_values:
            muval = np.atleast_2d(muval)

            extended_front = np.concatenate((non_dominated_front, muval))
            last_included_point_index = len(extended_front) - 1
            ndf, dl, dc, ndr = fast_non_dominated_sorting(points=extended_front.tolist())

            if last_included_point_index in ndf[0]:
                dominated_boolean_vector.append(0) # 0 as non-dominated
                non_dominated_vectors.append(muval)

            else:

                dominated_boolean_vector.append(1) # 1 as dominated

            utility_val = self.get_posterior_utility_landscape(-muval.reshape(-1)).reshape(-1)
            utility_vector.append(utility_val)

        # P = self.model.get_Y_values()

        non_dominated_vectors = np.array(non_dominated_vectors).squeeze()

        # plt.scatter(-generated_mu_values[:, 0], -generated_mu_values[:, 1], color="red")
        # plt.scatter(-np.array(non_dominated_vectors)[:,0],-np.array(non_dominated_vectors)[:,1], color="yellow" )
        # plt.scatter(-non_dominated_front[:, 0], -non_dominated_front[:, 1], color="blue")
        # plt.scatter(P[0], P[1], color="black")
        # plt.show()

        return np.array(non_dominated_vectors)

    def generate_mu_values(self):
        X = initial_design('latin',self.space, 500)
        generated_mu_values = self.model.posterior_mean(X)
        generated_var_values = self.model.posterior_variance(X, noise=False)

        y_lcb = generated_mu_values + self.alpha * np.sqrt(generated_var_values)
        y_lcb = np.transpose(y_lcb)
        return y_lcb

    def hypervolume_computation(self, non_dominated_surface,
                                newy):

        newy = np.array(newy).reshape(-1)
        utility_vector = []

        if self.verbose:
            P = self.model.get_Y_values()

            # non_dominated_vectors = np.array(non_dominated_vectors).squeeze()

            # plt.scatter(-generated_mu_values[:, 0], -generated_mu_values[:, 1], color="red")
            plt.scatter(-np.array(non_dominated_surface)[:,0],-np.array(non_dominated_surface)[:,1], color="yellow" )
            # plt.scatter(-non_dominated_front[:, 0], -non_dominated_front[:, 1], color="blue")

            for ndpoint in non_dominated_surface:
                surface_point = ndpoint.reshape(-1)

                if pareto_dominance(obj1=newy.tolist(), obj2=surface_point.tolist()):

                    utility_val = self.get_posterior_utility_landscape(-surface_point)
                    utility_vector.append(utility_val)
                else:
                    utility_vector.append(0)
            plt.scatter(-non_dominated_surface[:,0], -non_dominated_surface[:,1],
                        c=np.array(utility_vector).reshape(-1))
            # print(newy)
            plt.scatter(-newy[0], -newy[1], color="magenta")
            plt.scatter(P[0], P[1], color="red")
            plt.show()

        else:
            for ndpoint in non_dominated_surface:
                surface_point = ndpoint.reshape(-1)

                if pareto_dominance(obj1=newy.tolist(), obj2=surface_point.tolist()):
                    utility_val = self.get_posterior_utility_landscape(-surface_point)
                    utility_vector.append(utility_val)

        if len(utility_vector)==0:

            Y = self.model.get_Y_values()
            Y = np.concatenate(Y, axis=1)

            newy = np.atleast_2d(newy)
            sampled_pareto_solutions = np.concatenate((newy, -Y))
            ndf, dl, dc, ndr = fast_non_dominated_sorting(points=sampled_pareto_solutions.tolist())
            non_dominated_indeces = ndf[0]

            if 0 in non_dominated_indeces:

                hypervolume_value = self.get_posterior_utility_landscape(-newy)
                hypervolume_value = hypervolume_value / (len(non_dominated_surface) + 1)

            else:
                hypervolume_value = 0

        else:
            utility_val_newy = self.get_posterior_utility_landscape(-newy)
            utility_vector.append(utility_val_newy)
            hypervolume_value = np.sum(utility_vector)/(len(non_dominated_surface)+1)
        return hypervolume_value


    def compute_HVI_LCB(self, X):
        X = np.atleast_2d(X)

        P = self.model.get_Y_values()
        P_cur = (-np.concatenate(P,axis=1)).tolist()

        non_dominated_surface = self.initial_filtering_hypervolume(P_cur)
        # print(non_dominated_surface)
        non_dominated_surface = non_dominated_surface.squeeze()
        # print("non_dom_surface", non_dominated_surface)
        # print("-self.PF",-self.PF)
        # non_dominated_surface = np.concatenate((non_dominated_surface, -self.PF))

        # if self.verbose:
        #     plot_P_cur = np.array(P_cur)
        #     plt.scatter(plot_P_cur[:,0], plot_P_cur[:,1])
        #     non_dominated_surface = non_dominated_surface.squeeze()
        #     plt.scatter(non_dominated_surface[:,0],non_dominated_surface[:,1], color="red" )
        #
        #     mu_recommended = -self.model.posterior_mean(X)
        #     plt.scatter(mu_recommended[0], mu_recommended[1],color="magenta")
        #     plt.show()

        HVvals = np.zeros(X.shape[0])

        for i in range(len(X)):
            x = np.atleast_2d(X[i])
            ## Hypervolume computations
            mu = -self.model.posterior_mean(x)
            var = self.model.posterior_variance(x, noise=False)

            y_lcb = mu - self.alpha * np.sqrt(var)
            y_lcb = np.transpose(y_lcb)
            P_new = (-np.concatenate(P,axis=1)).tolist()


            HV_new = self.hypervolume_computation(non_dominated_surface=non_dominated_surface,
                                                  newy=y_lcb)

            HVvals[i] = HV_new


        return  HVvals

    def Generate_Pareto_Front(self):
        X_train = self.model.get_X_values()
        GP_y_predictions  = self.mean_prediction_model(X_train ,self.model)


        bounds = self.space.get_continuous_bounds()
        bounds = self.bounds_format_adapter(bounds)
        udp = GA(f=GP_y_predictions, bounds=bounds, n_obj=self.output_dimensions)
        pop = population(prob=udp, size=100)
        algo = algorithm(nsga2(gen=300))
        pop = algo.evolve(pop)
        fits, vectors = pop.get_f(), pop.get_x()
        ndf, dl, dc, ndr = fast_non_dominated_sorting(fits)
        result_x = vectors[ndf[0]]
        result_fx = fits[ndf[0]]

        return -result_fx, result_x  # GP_y_predictions

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
