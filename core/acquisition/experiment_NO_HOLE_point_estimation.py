import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import HOLE, NO_HOLE
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from ParEGO_acquisition import ParEGO
from bayesian_optimisation import BO
import os
from DecisionMakerLastStepsInteraction import AcquisitionFunctionandDecisionMakerInteraction
from point_estimation_EI_UU_acquisition import ExpectedImprovementUtilityUncertaintywithPointUtility
from utility_core import *
#ALWAYS check cost in
# --- Function to optimize


def point_estimation_NO_HOLE_function_caller_test(rep):

    rep = rep + 20
    noise = 1e-6
    np.random.seed(rep)


    max_number_DMqueries = [1]
    first_query_iteration = [[0, 1 , 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]]

    for num_queries_idx in range(len(max_number_DMqueries)):

        for first_query_iteration_element in first_query_iteration[num_queries_idx]:

            folder = "RESULTS"
            subfolder = "NO_HOLE_point_estimation_Assum_Lin_U_Lin_n_queries_" + str(max_number_DMqueries[num_queries_idx])+"_first_iteration_"+str(first_query_iteration_element)
            cwd = os.getcwd()
            path = cwd + "/" + folder + "/"+subfolder

            # include function
            func= NO_HOLE(sd=np.sqrt(noise))

            # --- Attributes
            #repeat same objective function to solve a 1 objective problem
            f = MultiObjective([func.f1, func.f2])

            # --- Attributes
            #repeat same objective function to solve a 1 objective problem

            # --- Space
            #define space of variables
            space =  GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-1.0, 1.0)},
                                                 {'name': 'var_2', 'type': 'continuous', 'domain': (-1.0, 1.0)}])
            n_f = 2
            input_d = 2

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

            assumed_u_funcs = [Lin_u]
            BayesInferenceUtility = Inference_method(assumed_u_funcs)

            # #Utility of the decision maker
            # Lin_u = Linear_utility_func(n_params=n_f)
            # Tche_u = Tchevichev_utility_func(n_params=n_f)
            # true_u_funcs = [Lin_u]

            # --- Utility function
            EI_UU = ExpectedImprovementUtilityUncertaintywithPointUtility(model=model_f,
                                                          space=space,
                                                          optimizer = acq_opt,
                                                          Inference_Object=BayesInferenceUtility)



            evaluator = GPyOpt.core.evaluators.Sequential(EI_UU)

            # --- Decision Maker interaction with the Front Class

            u_funcs_true = [Lin_u]
            InteractionwithDecisionMakerClass = ParetoFrontGeneration(model=model_f,
                                                                      space=space,
                                                                      seed=rep,
                                                                      utility=u_funcs_true)

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
#  function_caller_test_function_2_penalty(rep)
# for rep in range(10):
# point_estimation_NO_HOLE_function_caller_test(1)
# print("ready")


