import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import NO_HOLE
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from continuous_KG import KG
from ParEGO_acquisition import ParEGO
#from Constrained_U_KG import AcquisitionUKG
# from U_KG import AcquisitionUKG
from bayesian_optimisation_variable_Last_Step_exp import BO
import pandas as pd
import os

from parameter_distribution import ParameterDistribution
from utility import Utility
from scipy.stats import dirichlet

from BO_last_step_Ushape_PLOTS import Last_Step
from time import time as time
#ALWAYS check cost in
# --- Function to optimize


def NO_HOLE_function_caller_test(rep):
    penalty = 0
    noise = 1e-6
    alpha = 1.95

    n_f = 1
    n_c = 0
    input_d = 2
    m = 2
    Overall_Budget = 40
    for LS_budget in [0, 1, 3, 9,  15,  21,  27,  33,  39]:
        np.random.seed(rep)
        Last_Step_Budget = LS_budget
        Main_Alg_Budget = Overall_Budget - Last_Step_Budget
        n_initial_design = 2 * (input_d + 1)

        folder = "RESULTS"
        subfolder = "NO_HOLE_ParEGO_Main_" + str(Main_Alg_Budget) + "_" + "Last_Step_Budget_" + str(Last_Step_Budget) +"_Total Budget_"+str(Overall_Budget )
        cwd = os.getcwd()
        path = cwd + "/" + folder + "/" + subfolder

        # func2 = dropwave()
        POL_func= NO_HOLE(sd=np.sqrt(noise))
        ref_point = POL_func.ref_point

        # --- Attributes
        #repeat same objective function to solve a 1 objective problem
        f = MultiObjective([POL_func.f1, POL_func.f2])
        # c = MultiObjective([POL_func.c1, POL_func.c2])

        # --- Attributes
        # repeat same objective function to solve a 1 objective problem

        # c2 = MultiObjective([test_c2])
        # --- Space
        # define space of variables
        space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (-1.0, 1.0)},
                                           {'name': 'var_2', 'type': 'continuous', 'domain': (-1.0,
                                                                                              1.0)}])  # GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#

        model_f = multi_outputGP(output_dim=n_f, noise_var=[noise] * n_f, exact_feval=[True] * n_f)
        # model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-7]*n_c, exact_feval=[True]*n_c)

        # --- Aquisition optimizer
        # optimizer for inner acquisition function
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', inner_optimizer='Nelder_Mead',
                                                           space=space, model=model_f, model_c=None)

        # --- Initial design
        # initial design
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, n_initial_design)

        # --- Utility function
        def prior_sample_generator(n_samples=1, seed=None):
            if seed is None:

                samples = np.random.dirichlet(np.ones((m,)), n_samples)
                print("samples", samples)

            else:
                random_state = np.random.RandomState(seed)
                samples = random_state.dirichlet(np.ones((m,)), n_samples)
                print("samples", samples)

            return samples

        def prior_density(x):
            assert x.shape[1] == m, "wrong dimension"
            output = np.zeros(x.shape[0])
            for i in range(len(output)):
                output[i] = dirichlet.pdf(x=x[i], alpha=np.ones((m,)))
            return output.reshape(-1)

        def U_func(parameter, y):
            w = parameter
            scaled_vectors = np.multiply(w, y)
            utility = np.max(scaled_vectors, axis=1)
            utility = np.atleast_2d(utility)
            print("utility", utility.T)
            return utility.T

        def dU_func(parameter, y):
            raise
            return 0

        ##### Utility
        n_samples = 1
        support = prior_sample_generator(
            n_samples=n_samples)  # generates the support to marginalise the parameters for acquisition function inside the optimisation process. E_theta[EI(x)]

        prob_dist = prior_density(support)  # generates the initial density given the the support
        prob_dist /= np.sum(prob_dist)
        parameter_distribution = ParameterDistribution(continuous=True, support=support, prob_dist=prob_dist,
                                                       sample_generator=prior_sample_generator)

        U = Utility(func=U_func, dfunc=dU_func, parameter_dist=parameter_distribution, linear=True)

        # acquisition = HVI(model=model_f, model_c=model_c , alpha=alpha, space=space, optimizer = acq_opt)
        acquisition = ParEGO(model=model_f, model_c=None, alpha=alpha, space=space, NSGA_based=False, optimizer=acq_opt,
                             utility=U, true_func=f)

        weight = prior_sample_generator(n_samples=1, seed=rep)
        last_step_acquisition = Last_Step(model_f=model_f, model_c=None, true_f=f, true_c=None, n_f=m, n_c=n_c,
                                          Overall_Budget=Main_Alg_Budget + n_initial_design - 1, B=Last_Step_Budget,
                                          acquisition_optimiser=acq_opt, acquisition_f=acquisition, seed=rep,
                                          weight=weight, space=space, path=path)

        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, None, space, f, None, acquisition, evaluator, initial_design, ref_point=ref_point)

        # print("Finished Initialization")
        X, Y, C, Opportunity_cost = bo.run_optimization(max_iter=Main_Alg_Budget, rep=rep,
                                                        last_step_evaluator=last_step_acquisition, path=path,
                                                        verbosity=False)
        print("Code Ended")

        # data = {}
        # data["Opportunity_cost"] = np.array(Opportunity_cost).reshape(-1)
        #
        # gen_file = pd.DataFrame.from_dict(data)
        # folder = "RESULTS"
        # subfolder = "DEB_HVI_"
        # cwd = os.getcwd()
        # print("cwd", cwd)
        # path = cwd + "/" + folder +"/"+ subfolder +'/it_' + str(rep)+ '.csv'
        # if os.path.isdir(cwd + "/" + folder +"/"+ subfolder) == False:
        #     os.makedirs(cwd + "/" + folder +"/"+ subfolder)
        #
        # gen_file.to_csv(path_or_buf=path)

        print("X", X, "Y", Y, "C", C)

# for rep in range(10):
#  function_caller_test_function_2_penalty(rep)
# NO_HOLE_function_caller_test(rep=99)
print("ready")


