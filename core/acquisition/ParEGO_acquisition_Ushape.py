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


class ParEGO(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details. 
    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, model_c=None, alpha=1.95,optimizer=None, NSGA_based=False,cost_withGradients=None, utility=None, true_func=None):
        self.optimizer = optimizer
        self.utility = utility
        self.MCMC = False

        self.counter = 0
        self.current_max_value = np.inf
        self.Z_samples_obj = None
        self.Z_samples_const = None
        self.true_func = true_func
        self.saved_Nx = -10
        self.name = "Constrained_HVI"
        if model_c is None:
            self.constraints_flag = False
        else:
            self.constraints_flag = True
        # print(alpha)
        # print(cost_withGradients)
        # print(model_c)
        super(ParEGO, self).__init__(model, space, optimizer, model_c = model_c, alpha =alpha, NSGA_based=NSGA_based,cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients


    def _compute_acq(self, X, ref_point=None, starting_point=None):
        """
        Computes the aquisition function
        
        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        # print("_compute_acq")

        X =np.atleast_2d(X)
        acq_function_val = self.expected_improvement(X)
        return np.array(acq_function_val).reshape(-1)



    def expected_improvement(self, X, offset=1e-4):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        '''

        #print("DETERMINISTIC LAST STEP")
        mu = -self.model.posterior_mean(X)
        var = self.model.posterior_variance(X, noise=False)

        sigma = np.sqrt(var).reshape(-1, 1)
        mu = mu.reshape(-1,1)

        mu_sample_opt = -np.min(self.model.get_Y_values())

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        # print("ei", ei)
        return np.array(ei).reshape(-1)

    def _compute_acq_withGradients(self, X):
        """
        """

        print("Gradients not Implemented")
        raise

        return 0