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
from numba import jit
from pathos.multiprocessing import ProcessingPool as Pool
from pyDOE import *

class ExpectedImprovementUtilityUncertainty(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details. 
    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, Inference_Object=None):

        #get needed functions
        self.optimizer = optimizer
        self.Inference_Object = Inference_Object
        self.output_dim = model.output_dim
        lhd = lhs(self.output_dim, samples=50)

        #initialise variables
        self.current_max_value = np.inf
        self.Z_samples_obj = None
        self.W_samples = norm(loc=0, scale=1).ppf(lhd)  # Pseudo-random number generation to compute expectation
        self.old_number_of_simulation_samples = 0
        self.number_of_gp_hyps_samples = 1
        self.n_samples = 50
        self.fantasised_posterior_samples = None

        super(ExpectedImprovementUtilityUncertainty, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients

        self.utility = self.Inference_Object.get_utility_function()
        self.probability_models, _, self.models = self.Inference_Object.get_utility_information()

    def include_fantasised_posterior_samples(self, posterior_samples):
        self.fantasised_posterior_samples = posterior_samples

    def get_posterior_samples(self, n_samples=None):
        # posterior_samples = self.Inference_Object.get_generated_posterior_samples()
        # if posterior_samples is None:
        if n_samples is None:
            posterior_samples = self.Inference_Object.posterior_sampler(self.n_samples)
        else:
            posterior_samples = self.Inference_Object.posterior_sampler(n_samples)
        return posterior_samples

    def _compute_acq(self, X, parallel=True):
        """
        Computes the aquisition function

        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        X = np.atleast_2d(X)
        current_number_simulation_points = self.model.get_X_values().shape[0]
        if not self.old_number_of_simulation_samples == current_number_simulation_points:
            print("Recomputing Z vectors montecarlo...")
            lhd = lhs(self.model.output_dim, samples=50)
            self.W_samples = norm(loc=0, scale=1).ppf(lhd)  # Pseudo-random number generation to compute expectation
            self.old_number_of_simulation_samples = current_number_simulation_points
            # self.posterior_samples = self.get_posterior_samples(10)
            # print(self.posterior_samples)
            # raise
            self.probability_models, self.posterior_parameter_samples, self.models = self.Inference_Object.get_utility_information()

        marginal_acqX = self._marginal_acq(X)
        acqX = marginal_acqX  # acqX = np.sum(marginal_acqX, axis=1) / len(self.utility_parameter_samples)

        #acqX = np.reshape(acqX, (X.shape[0], 1))
        return acqX.reshape(-1)

    def _marginal_acq(self, X):
        """
        """

        marginal_acqX = np.zeros((X.shape[0], self.number_of_gp_hyps_samples))
        Z = self.W_samples[np.newaxis, :, np.newaxis, :]

        # print("linweight",linear_weight_combination.shape)
        # ALL dimensions adapated to be (Ntheta, Nz, Nx, Dimy)
        for h in range(self.number_of_gp_hyps_samples):
            # self.model.set_hyperparameters(h)
            EI_val_model = []
            # print("models", self.models)
            for utility_idx, utility in enumerate(self.models):
                # print("utility, utility_idx",utility, utility_idx)
                meanX, varX = self.model.predict(X)
                MU = np.stack(meanX, axis=1)
                VARX = np.stack(varX, axis=1)
                sigmaX = np.sqrt(VARX)

                # best utility samples so far
                fX_evaluated = self.model.posterior_mean_at_evaluated_points()
                fX_evaluated = np.stack(fX_evaluated, axis=1)[np.newaxis, np.newaxis, :, :]

                # print("self.posterior_parameter_samples[utility_idx].shape[0]",self.posterior_parameter_samples)
                Best_Sampled_Utility =utility(y = fX_evaluated,
                                              weights=np.ones((self.posterior_parameter_samples[utility_idx][0].shape[0], 1)),
                                              parameters=self.posterior_parameter_samples[utility_idx],
                                              vectorised=True)
                # print("Best_Sampled_Utility",Best_Sampled_Utility.shape)
                max_valX_evaluated = np.max(Best_Sampled_Utility, axis=-1)
                max_valX_evaluated = max_valX_evaluated[:, :, np.newaxis]
                # print("max_valX_evaluated",max_valX_evaluated.shape)
                MU = MU[np.newaxis, np.newaxis, :, :]
                sigmaX = sigmaX[np.newaxis, np.newaxis, :, :]
                Y = MU + sigmaX * Z

                # print("Y", Y.shape)
                Utility = utility(y = Y,
                                  weights=np.ones((self.posterior_parameter_samples[utility_idx][0].shape[0], 1)),
                                  parameters=self.posterior_parameter_samples[utility_idx],
                                  vectorised=True)


                Improvement = Utility - max_valX_evaluated
                Improvement[Improvement < 0] = 0.0

                # print("Improvement", np.mean(Improvement, axis=(0, 1)))
                marginal_acqX[:, h] += np.mean(Improvement, axis=(0, 1)) * self.probability_models[utility_idx]
                # EI_val_model.append(marginal_acqX[:, h])

            overall_EI_val = marginal_acqX #np.sum([EI_val_model[m]  for m in range(len(self.models))], axis=0)
            # print(self.probability_models, self.models)
            # print("X",X.shape)
            # print("overall_EI_val",overall_EI_val.shape)
            # raise
        # marginal_acqX = np.sum(marginal_acqX, axis=1)
        overall_EI_val /= self.number_of_gp_hyps_samples

        return overall_EI_val


    def _compute_acq_withGradients(self, X):
        """
        """

        raise
        return 0

    def _marginal_acq_with_gradient(self, X, utility_parameter_samples):
        raise
        return 0

