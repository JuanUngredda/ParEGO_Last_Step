import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import DEB
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from continuous_KG import KG
from constrained_HVI_acquisition import HVI
from Constrained_U_KG import AcquisitionUKG
# from Last_Step_U_KG import AcquisitionUKG as Last_Step_KG
# from U_KG import AcquisitionUKG
from bayesian_optimisation import BO
import pandas as pd
import os

from parameter_distribution import ParameterDistribution
from utility import Utility
from scipy.stats import dirichlet
from BO_last_step import Last_Step
from time import time as time
#ALWAYS check cost in
# --- Function to optimize


def DEB_function_caller_test(rep):

    penalty =0
    noise = 1e-3
    alpha =1.95
    np.random.seed(rep)
    folder = "RESULTS"
    subfolder = "DEB_KG"
    cwd = os.getcwd()
    path = cwd + "/" + folder + "/"+subfolder
    # func2 = dropwave()
    DEB_func= DEB(sd=np.sqrt(noise))
    ref_point = DEB_func.ref_point

    # --- Attributes
    #repeat same objective function to solve a 1 objective problem
    f = MultiObjective([DEB_func.f1, DEB_func.f2])
    c = MultiObjective([DEB_func.c1, DEB_func.c2])

    # --- Attributes
    #repeat same objective function to solve a 1 objective problem

    #c2 = MultiObjective([test_c2])
    # --- Space
    #define space of variables
    space =  GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 1.0)},{'name': 'var_2', 'type': 'continuous', 'domain': (0, 5.0)}])#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#
    n_f = 2
    n_c = 2
    input_d = 2
    m =n_f

    model_f = multi_outputGP(output_dim = n_f,   noise_var=[noise]*n_f, exact_feval=[True]*n_f)
    model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-6]*n_c, exact_feval=[True]*n_c)

    # --- Aquisition optimizer
    #optimizer for inner acquisition function
    acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space, model=model_f, model_c=model_c,NSGA_based=False,analytical_gradient_prediction=True)

    # --- Initial design
    #initial design
    initial_design = GPyOpt.experiment_design.initial_design('latin', space, 20)#2*(input_d+1))

    # --- Utility function
    def prior_sample_generator(n_samples=1, seed=None):
        if seed is None:
            samples = np.random.dirichlet(np.ones((m,)), n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.dirichlet(np.ones((m,)), n_samples)
        return samples

    def prior_density(x):
        assert x.shape[1] == m, "wrong dimension"
        output = np.zeros(x.shape[0])
        for i in range(len(output)):
            output[i] = dirichlet.pdf(x=x[i], alpha=np.ones((m,)))
        return output.reshape(-1)

    def U_func(parameter, y):
        return np.dot(parameter, y)

    def dU_func(parameter, y):
        return parameter

    ##### Utility
    n_samples = 5
    support = prior_sample_generator(n_samples=n_samples)  # generates the support to marginalise the parameters for acquisition function inside the optimisation process. E_theta[EI(x)]

    prob_dist = prior_density(support)  # generates the initial density given the the support
    prob_dist /= np.sum(prob_dist)
    parameter_distribution = ParameterDistribution(continuous=True, sample_generator=prior_sample_generator)


    U = Utility(func=U_func, dfunc=dU_func, parameter_dist=parameter_distribution, linear=True)


    #acquisition = HVI(model=model_f, model_c=model_c , alpha=alpha, space=space, optimizer = acq_opt)
    acquisition = AcquisitionUKG(model=model_f, model_c=model_c , alpha=alpha, space=space, optimizer = acq_opt, utility= U, true_func=f)
    last_step_acquisition = Last_Step(model_f=model_f, model_c=model_c , true_f=f, true_c=c,n_f=n_f, n_c=n_c, acquisition_optimiser = acq_opt, seed=rep, path=path)


    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    #GPyOpt.core.evaluators.Sequential(last_step_acquisition)
    bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design,  ref_point=ref_point)


    max_iter  = 25
    # print("Finished Initialization")
    X, Y, C, Opportunity_cost = bo.run_optimization(max_iter = max_iter, rep=rep, last_step_evaluator=last_step_acquisition, path=path, verbosity=True)
    print("Code Ended")



    print("X",X,"Y",Y, "C", C)

# for rep in range(10):
#     function_caller_test_function_2_penalty(rep)
DEB_function_caller_test(rep=1)


