import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import HOLE, NO_HOLE
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from ParEGO_acquisition import ParEGO
from bayesian_optimisation import BO
import os
from DecisionMakerLastStepsInteraction import AcquisitionFunctionandDecisionMakerInteraction
from EI_UU_acquisition import ExpectedImprovementUtilityUncertainty
from utility_core import *
import torch
from botorch.test_functions.multi_objective import VehicleSafety, ZDT1

d = 3
m = 2
dtype = torch.double
fun = ZDT1(dim=d,negate=False).to(
    dtype=dtype
)

space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

def f1(X, true_val=None):
    X = torch.Tensor(X)
    fval = fun(X)[:, 0].numpy()
    return -fval
def f2(X, true_val=None):
    X = torch.Tensor(X)
    fval = fun(X)[:, 1].numpy()
    return -fval

#ALWAYS check cost in
# --- Function to optimize


def ZDT1_function_caller_test(rep):

    rep= rep #+ 20
    noise = 1e-6
    np.random.seed(rep)


    max_number_DMqueries = [ 5]
    first_query_iteration = [[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]]

    for num_queries_idx in range(len(max_number_DMqueries)):

        for first_query_iteration_element in first_query_iteration[num_queries_idx]:

            folder = "RESULTS"
            subfolder = "ZDT1_region_pick_Bayes_Assum_Tche_U_Tche_n_queries_" + str(max_number_DMqueries[num_queries_idx])+"_first_iteration_"+str(first_query_iteration_element)
            cwd = os.getcwd()
            path = cwd + "/" + folder + "/"+subfolder

            # include function
            # func= NO_HOLE(sd=np.sqrt(noise))

            # --- Attributes
            #repeat same objective function to solve a 1 objective problem
            f = MultiObjective([f1, f2])

            # --- Attributes
            #repeat same objective function to solve a 1 objective problem

            # --- Space
            #define space of variables
            space = GPyOpt.Design_space(
                space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])
            n_f = m
            input_d = d

            model_f = multi_outputGP(output_dim = n_f,
                                     noise_var=[noise]*n_f,
                                     exact_feval=[True]*n_f)

            # --- Aquisition optimizer
            #optimizer for inner acquisition function
            acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs',
                                                               inner_optimizer='Nelder_Mead',
                                                               space=space,
                                                               model=model_f)


            # --- Initial design
            #initial design
            initial_design = GPyOpt.experiment_design.initial_design('latin',
                                                                     space, 2*(input_d+1))


            # --- Bayesian Inference Object on the Utility

            #utility functions assumed for the decision maker

            Tche_u = Tchevichev_utility_func(n_params=n_f)
            Lin_u = Linear_utility_func(n_params=n_f)

            assumed_u_funcs = [Tche_u]
            BayesInferenceUtility = Inference_method(assumed_u_funcs)

            # #Utility of the decision maker
            # Lin_u = Linear_utility_func(n_params=n_f)
            # Tche_u = Tchevichev_utility_func(n_params=n_f)
            # true_u_funcs = [Lin_u]

            # --- Utility function
            EI_UU = ExpectedImprovementUtilityUncertainty(model=model_f,
                                                          space=space,
                                                          optimizer = acq_opt,
                                                          Inference_Object=BayesInferenceUtility)

            # --- Decision Maker interaction with the Front Class

            #utility functions assumed for the decision maker

            u_funcs_true = [Tche_u]
            InteractionwithDecisionMakerClass = ParetoFrontGeneration(model=model_f,
                                                                      space=space,
                                                                      seed=rep,
                                                                      utility=u_funcs_true)


            evaluator = GPyOpt.core.evaluators.Sequential(EI_UU)



            AcquisitionwithDMInteration = AcquisitionFunctionandDecisionMakerInteraction(model=model_f,
                                                                                         true_f=f,
                                                                                         space=space,
                                                                                         acquisition_f=EI_UU,
                                                                                         acquisition_optimiser=acq_opt,
                                                                                         InteractionClass = InteractionwithDecisionMakerClass,
                                                                                         Inference_Object=BayesInferenceUtility,
                                                                                         NLastSteps=2,
                                                                                         path=path,
                                                                                         seed=rep)

            bo = BO(model=model_f,
                    space=space,
                    acquisition=EI_UU,
                    objective=f,
                    evaluator=evaluator,
                    X_init=initial_design,
                    DecisionMakerInteractor = AcquisitionwithDMInteration)


            # print("Finished Initialization")
            X, Y, Opportunity_cost = bo.run_optimization(max_iter =100,
                                                            rep=rep,
                                                            path=path,
                                                            verbosity=False,
                                                             max_number_DMqueries=max_number_DMqueries[num_queries_idx],
                                                             first_query_iteration=first_query_iteration_element
                                                             )

        print("Code Ended")

        print("X",X,"Y",Y)

# for rep in range(10):
# HOLE_function_caller_test(rep=1)
# for rep in range(10):
# ZDT1_function_caller_test(3)
# print("ready")

