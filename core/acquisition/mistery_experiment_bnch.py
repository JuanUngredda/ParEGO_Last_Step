import numpy as np
import GPyOpt
from GPyOpt.objective_examples.experiments2d import mistery,  test_function_2, new_brannin
import GPy as GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
import matplotlib.pyplot as plt
import scipy
from continuous_KG import KG
from bayesian_optimisation_benchmark import BO
import pandas as pd
import os
from time import time as time
#ALWAYS check cost in
# --- Function to optimize

def function_caller_test_function_2_penalty(rep):
    for penalty in [-999, 4]:

        noise = 1e-6
        np.random.seed(rep)

        # func2 = dropwave()
        mistery_f =new_brannin(sd=np.sqrt(noise))

        # --- Attributes
        #repeat same objective function to solve a 1 objective problem
        f = MultiObjective([mistery_f.f, mistery_f.f])
        c = MultiObjective([mistery_f.c])

        # --- Attributes
        #repeat same objective function to solve a 1 objective problem

        #c2 = MultiObjective([test_c2])
        # --- Space
        #define space of variables
        space =  GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},{'name': 'var_2', 'type': 'continuous', 'domain': (0,15)}])#GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,100)}])#
        n_f = 1
        n_c = 1
        model_f = multi_outputGP(output_dim = n_f,   noise_var=[noise]*n_f, exact_feval=[True]*n_f)
        model_c = multi_outputGP(output_dim = n_c,  noise_var=[1e-21]*n_c, exact_feval=[True]*n_c)


        # --- Aquisition optimizer
        #optimizer for inner acquisition function
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs', space=space, model=model_f, model_c=model_c)
        #
        # # --- Initial design
        #initial design
        initial_design = GPyOpt.experiment_design.initial_design('latin', space, 10)


        acquisition = KG(model=model_f, model_c=model_c , space=space, optimizer = acq_opt)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = BO(model_f, model_c, space, f, c, acquisition, evaluator, initial_design, penalty_tag="fixed", penalty_value=penalty)


        max_iter  = 35
        # print("Finished Initialization")
        X, Y, C, Opportunity_cost = bo.run_optimization(max_iter = max_iter,verbosity=False)
        print("Code Ended")

        data = {}
        data["Opportunity_cost"] = np.array(Opportunity_cost).reshape(-1)

        gen_file = pd.DataFrame.from_dict(data)
        folder = "RESULTS"
        subfolder = "new_brannin_proposed_" +str(penalty)
        cwd = os.getcwd()
        print("cwd", cwd)
        path = cwd + "/" + folder +"/"+ subfolder +'/it_' + str(rep)+ '.csv'
        if os.path.isdir(cwd + "/" + folder +"/"+ subfolder) == False:
            os.makedirs(cwd + "/" + folder +"/"+ subfolder)

        gen_file.to_csv(path_or_buf=path)

        print("X",X,"Y",Y, "C", C)

# for rep in range(10):
#     function_caller_test_function_2_penalty(rep)
#function_caller_mistery_bnch(rep=21)


