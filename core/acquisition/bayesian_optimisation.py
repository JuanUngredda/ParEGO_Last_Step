# Copyright (c) 2018, Raul Astudillo

import GPyOpt
import collections
import numpy as np
#import pygmo as pg
import time
import csv
import matplotlib.pyplot as plt
from pyDOE import lhs
import time
from GPyOpt.DM.Decision_Maker import DM
from GPyOpt.DM.inference import inf
from GPyOpt.experiment_design import initial_design
from GPyOpt.util.general import best_value
from GPyOpt.util.duplicate_manager import DuplicateManager
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.cost import CostModel
from GPyOpt.optimization.acquisition_optimizer import ContextManager
from scipy.stats import norm
from pygmo import hypervolume
import pandas as pd
import os
try:
    from GPyOpt.plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass


class BO(object):
    """
    Runner of the multi-attribute Bayesian optimization loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: GPyOpt DuplicateManager class. Avoids re-evaluating the objective at previous, pending or infeasible locations (default, False).
    """


    def __init__(self, model,
                 space,
                 objective,
                 evaluator,
                 acquisition,
                 X_init ,
                 expensive=False,
                 Y_init=None,
                 model_update_interval = 1,
                 deterministic=True,
                 DecisionMakerInteractor  = None):

        self.acquisition = acquisition
        self.model = model
        self.space = space
        self.objective = objective
        self.evaluator = evaluator
        self.model_update_interval = model_update_interval
        self.X = X_init
        self.Y = Y_init
        self.deterministic = deterministic
        self.model_parameters_iterations = None
        self.expensive = expensive
        self.DecisionMakerInteractor =DecisionMakerInteractor

        print("name of acquisition function wasnt provided")
        self.sample_from_acq = False
        self.tag_last_evaluation = False

    def suggest_next_locations(self, context = None, pending_X = None, ignored_X = None):
        """
        Run a single optimization step and return the next locations to evaluate the objective.
        Number of suggested locations equals to batch_size.

        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param pending_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet) (default, None).
        :param ignored_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again (default, None).
        """
        self.model_parameters_iterations = None
        self.num_acquisitions = 0
        self.context = context
        self._update_model(self.normalization_type)

        suggested_locations = self._compute_next_evaluations(pending_zipped_X=pending_X, ignored_zipped_X=ignored_X)
        return suggested_locations
    

    def run_optimization(self, max_iter = 1,
                         max_time = np.inf,
                         rep = None ,
                         eps = 1e-8,
                         context = None,
                         verbosity=False,
                         path=None,
                         evaluations_file = None,
                         max_number_DMqueries=0,
                         first_query_iteration=0):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param evaluations_file: filename of the file where the evaluated points and corresponding evaluations are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.evaluations_file = evaluations_file
        self.context = context
        self.path =path
        self.rep = rep
        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)


        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y
        self.Opportunity_Cost = {"Hypervolume": np.array([])}
        self.data = {}

        self.data["Utility_sampled"] = np.array([])
        self.data["Utility"] = np.array([])
        self.data["Best_Utility"] = np.array([])
        value_so_far = []

        # --- Initialize time cost of the evaluations
        print("MAIN LOOP STARTS")
        self.true_best_stats = {"true_best": [], "mean_gp": [], "std gp": [], "pf": [], "mu_pf": [], "var_pf": [],
                                "residual_noise": []}
        self._update_model()

        #Decision maker's variable initialisation and values
        Pareto_fronts = []
        picked_solution_indeces = []
        self.number_of_queries_taken = 0
        query_schedule = np.arange(first_query_iteration, first_query_iteration + max_number_DMqueries)
        while (self.max_iter > self.num_acquisitions ):

            # self.optimize_final_evaluation()
            print("maKG optimizer")
            start = time.time()

            # true_best_x, true_best_val = self.compute_underlying_best()
            #
            # design_plot = initial_design('random', self.space, 10000)
            #
            # func_val, _ = self.objective.evaluate(design_plot)
            # func_val_recommended ,_= self.objective.evaluate(true_best_x)
            # print(func_val_recommended)
            # plt.scatter(func_val[0], func_val[1])
            # plt.scatter(func_val_recommended [0], func_val_recommended [1])
            # plt.show()
            # print(true_best_val)
            # raise
            if (self.DecisionMakerInteractor is not None) \
                    and (self.num_acquisitions in query_schedule):

                #get data from decision maker
                solution_picked_dm_index, solution_x_picked_dm, PF, _ = self.DecisionMakerInteractor.get_DecisionMaker_data()

                #include new data
                Pareto_fronts.append(PF)
                picked_solution_indeces.append(solution_picked_dm_index)

                # Inference on pareto front and getting posterior samples
                posterior_samples = self.DecisionMakerInteractor.posterior_sample_generation(picked_solution_indeces,
                                                                                                Pareto_fronts)

                #Include new posterior samples in the acquisition function
                self.DecisionMakerInteractor.include_posterior_samples_in_acq(posterior_samples=posterior_samples)
                self.number_of_queries_taken+=1

            self.suggested_sample = self._compute_next_evaluations()
            print("self.suggested_sample",self.suggested_sample)


            if verbosity:
                self.verbosity_plot_2D_unconstrained()


            # self.suggested_sample = np.array([[0.1,0.0]])
            finish = time.time()
            print("time optimisation point X", finish - start)

            self.X = np.vstack((self.X,self.suggested_sample))
            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            self._update_model()
            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1
            print("optimize_final_evaluation")

            self.store_results(self.suggested_sample)

            print("self.X, self.Y, self.C , self.Opportunity_Cost",self.X, self.Y, self.Opportunity_Cost)

        return self.X, self.Y,  self.Opportunity_Cost
        # --- Print the desired result in files
        #if self.evaluations_file is not None:
            #self.save_evaluations(self.evaluations_file)

        #file = open('test_file.txt','w')
        #np.savetxt('test_file.txt',value_so_far)

    def verbosity_plot_1D(self):
        ####plots
        print("generating plots")
        design_plot = np.linspace(0,5,100)[:,None]

        # precision = []
        # for i in range(20):
        #     kg_f = -self.acquisition._compute_acq(design_plot)
        #     precision.append(np.array(kg_f).reshape(-1))

        # print("mean precision", np.mean(precision, axis=0), "std precision",  np.std(precision, axis=0), "max precision", np.max(precision, axis=0), "min precision",np.min(precision, axis=0))
        ac_f = self.expected_improvement(design_plot)

        Y, _ = self.objective.evaluate(design_plot)
        C, _ = self.constraint.evaluate(design_plot)
        pf = self.probability_feasibility_multi_gp(design_plot, self.model_c).reshape(-1, 1)
        mu_f = self.model.predict(design_plot)[0]

        bool_C = np.product(np.concatenate(C, axis=1) < 0, axis=1)
        func_val = Y * bool_C.reshape(-1, 1)

        kg_f = -self.acquisition._compute_acq(design_plot)
        design_plot = design_plot.reshape(-1)
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].set_title('True Function')
        axs[0, 0].plot(design_plot, np.array(func_val).reshape(-1))
        axs[0, 0].scatter(self.X, self.Y, color="red", label="sampled")
        # suggested_sample_value , _= self.objective.evaluate(self.suggested_sample)
        # axs[0, 0].scatter(self.suggested_sample, suggested_sample_value, marker="x", color="red",
        #                   label="suggested")

        axs[0, 0].legend()

        axs[0, 1].set_title('approximation Acqu Function')
        axs[0, 1].plot(design_plot, np.array(ac_f).reshape(-1))
        axs[0, 1].legend()

        axs[1, 0].set_title("mu and pf separetely ")
        axs[1, 0].plot(design_plot, np.array(mu_f).reshape(-1) , label="mu")
        axs[1, 0].plot(design_plot,  np.array(pf).reshape(-1), label="pf")
        axs[1, 0].legend()

        axs[1, 1].set_title("mu pf")
        axs[1, 1].plot(design_plot, np.array(mu_f).reshape(-1) * np.array(pf).reshape(-1))
        axs[1, 1].legend()

        axs[2, 1].set_title('approximation kg Function')
        axs[2, 1].plot(design_plot, np.array(kg_f).reshape(-1))
        axs[2, 1].legend()
        plt.show()

    def verbosity_plot_2D_unconstrained(self):
        ####plots
        print("generating plots")
        design_plot = initial_design('random', self.space, 10000)

        # precision = []
        # for i in range(20):
        # kg_f = -self.acquisition._compute_acq(design_plot)
        #     precision.append(np.array(kg_f).reshape(-1))

        # print("mean precision", np.mean(precision, axis=0), "std precision",  np.std(precision, axis=0), "max precision", np.max(precision, axis=0), "min precision",np.min(precision, axis=0))

        func_val, _ = self.objective.evaluate(design_plot)
        func_val = np.concatenate(func_val, axis=1)

        mu_f = self.model.posterior_mean(design_plot)
        mu_f = np.stack(mu_f, axis=1)
        var_f = self.model.posterior_variance(design_plot, noise=False)
        mu_predicted_best =self.model.posterior_mean(self.suggested_sample)
        mu_predicted_best = np.stack( mu_predicted_best, axis=1)

        # best_HVI = self.acquisition._compute_acq(self.suggested_sample)
        # raise

        HVI = self.acquisition._compute_acq(design_plot)

        # weighted_surface = self.acquisition.weighting_surface(design_plot)
        # true_underlying_utility = self.get_true_utility_function()
        # true_parameters = self.get_true_parameters()
        #
        # true_utility_values = true_underlying_utility(y = func_val,
        #                                            weights=true_parameters[1],
        #                                            parameters=true_parameters[0])
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].set_title('True PF Function')
        axs[0, 0].scatter(func_val[:, 0], func_val[:, 1])#, c=np.array(true_utility_values).reshape(-1) )

        Yvals = self.model.get_Y_values()
        Yvals = np.stack(Yvals, axis=1)

        # true_best_x, true_best_val = self.compute_underlying_best()
        # true_best_y, _ = self.objective.evaluate(true_best_x)

        # Pareto_front, preferred_points = self.acquisition.get_sampled_data()
        # Last_PF = Pareto_front[-1]
        # Last_preferred_point_idx = preferred_points[-1]
        #
        # preferred_point = Last_PF[Last_preferred_point_idx]

        axs[0, 1].set_title("GP(X)")
        # axs[0, 1].scatter(func_val[:, 0], func_val[:, 1], c=np.array(true_utility_values).reshape(-1))
        axs[0, 1].scatter(mu_f[:,0], mu_f[:,1], c= np.array(HVI).reshape(-1))
        # axs[0, 1].scatter(mu_f[:, 0], mu_f[:, 1], c=weighted_surface.reshape(-1))
        axs[0,1].scatter(mu_predicted_best[:,0],mu_predicted_best[:,1] , color="magenta")
        # axs[0,1].scatter(preferred_point[0], preferred_point[1], color="red")
        axs[0, 1].scatter(Yvals[:, 0], Yvals[:, 1], color="orange")
        # axs[0, 1].scatter(true_best_y[ 0], true_best_y[1], color="red")
        axs[0, 1].legend()

        posterior_samples = self.acquisition.get_posterior_samples()
        print("self.suggested_sample",self.suggested_sample)
        axs[1, 0].set_title("posterior samples $\Theta_{1}$")
        axs[1, 0].hist(posterior_samples[0][0][:,0])
        axs[1, 0].set_xlim([0, 1])

        axs[1, 1].set_title("posterior samples $\Theta_{2}$")
        axs[1, 1].hist(posterior_samples[0][0][:,1])
        axs[1, 1].set_xlim([0, 1])
        # axs[1, 1].set_title("acq(X)")
        # axs[1, 1].scatter(design_plot[:,0], design_plot[:,1], c= np.array(HVI).reshape(-1))
        # axs[1,1].scatter(self.suggested_sample[:,0],self.suggested_sample[:,1] , color="magenta")
        # axs[1, 1].legend()

        # axs[1, 1].set_title("mu pf")
        # axs[1, 1].scatter(design_plot[:,0],design_plot[:,1],c= np.array(mu_f).reshape(-1) * np.array(pf).reshape(-1))
        # axs[1, 1].legend()
        #
        # axs[1, 0].set_title('Opportunity Cost')
        # axs[1, 0].plot(range(len(self.Opportunity_Cost)), self.Opportunity_Cost)
        # axs[1, 0].set_yscale("log")
        # axs[1, 0].legend()

        # axs[1, 1].set_title('True PF Function with sampled points')
        # axs[1, 1].scatter(func_val[:, 0], func_val[:, 1])
        # axs[1, 1].scatter(self.Y[0],self.Y[1], color="red", label="sampled")

        # import os
        # folder = "IMAGES"
        # subfolder = "new_branin"
        # cwd = os.getcwd()
        # print("cwd", cwd)
        # time_taken = time.time()
        # path = cwd + "/" + folder + "/" + subfolder + '/im_' +str(time_taken) +str(self.X.shape[0]) + '.pdf'
        # if os.path.isdir(cwd + "/" + folder + "/" + subfolder) == False:
        #     os.makedirs(cwd + "/" + folder + "/" + subfolder)
        # plt.savefig(path)
        plt.show()


    def store_results(self, recommended_x):

        if self.DecisionMakerInteractor is None:

            P = self.model.get_Y_values()
            P_cur = (-np.concatenate(P, axis=1)).tolist()

            print("P_cur", P_cur)
            print("ref", self.acquisition.ref_point)
            HV0_func = hypervolume(P_cur)
            uval_sampled = HV0_func.compute(ref_point=self.acquisition.ref_point)

        else:
            true_underlying_utility = self.get_true_utility_function()
            true_parameters = self.get_true_parameters()


            Y_recommended, cost_new = self.objective.evaluate(recommended_x)
            Y_recommended = np.concatenate(Y_recommended, axis=1)
            true_recommended_utility = true_underlying_utility(y = Y_recommended,
                                           weights=true_parameters[1],
                                           parameters=true_parameters[0])

            out = true_recommended_utility.reshape(-1)

            self.data["Utility"] = np.concatenate((self.data["Utility"], np.array(out).reshape(-1)))

            Y_train = self.model.get_Y_values()
            Y_train = np.concatenate(Y_train, axis=1)


            uval_sampled = np.max(true_underlying_utility(y=Y_train,
                                                           weights=true_parameters[1],
                                                           parameters=true_parameters[0]))

            N_entries = len(self.data["Utility"].reshape(-1))
            true_best_x, true_best_val = self.compute_underlying_best()
            self.data["Best_Utility"] = np.repeat(true_best_val, N_entries)


        self.data["Utility_sampled"] = np.concatenate((self.data["Utility_sampled"], np.array(uval_sampled).reshape(-1)))


        if self.path is not None:
            gen_file = pd.DataFrame.from_dict(self.data)
            results_folder = "Hypervolume_improve"

            path = self.path +"/" + results_folder + '/it_' + str(self.rep) + '.csv'
            if os.path.isdir(self.path +"/" + results_folder  ) == False:
                os.makedirs(self.path +"/" + results_folder  , exist_ok=True)

            gen_file.to_csv(path_or_buf=path)

            # extra_data ={"parameters":true_parameters, "number_of_queries":self.number_of_queries_taken}
            # extra_gen_file = pd.DataFrame.from_dict(extra_data)
            # extra_path = self.path + "/" + results_folder + '/parameters_'+str(self.rep)+'.csv'
            # extra_gen_file.to_csv(path_or_buf=extra_path)

        print("path", self.path + "/" + results_folder)

    def get_true_utility_function(self):
        return self.DecisionMakerInteractor.get_true_utility_function()

    def get_true_parameters(self):
        return self.DecisionMakerInteractor.get_true_parameters()

    def get_optimiser(self):
        return self.DecisionMakerInteractor.get_optimiser()

    def compute_underlying_best(self):

        true_underlying_utility = self.get_true_utility_function()
        weight = self.get_true_parameters()
        optimiser =self.get_optimiser()

        def top_true_utility(X):
            X = np.atleast_2d(X)
            Y_recommended, cost_new = self.objective.evaluate(X)
            Y_recommended = np.concatenate(Y_recommended, axis=1)
            uval = true_underlying_utility(y = Y_recommended,
                                           weights= weight[1],
                                           parameters= weight[0])

            return -np.array(uval).reshape(-1)

        true_best_x, true_best_val = optimiser(f=top_true_utility)

        return true_best_x, -true_best_val


    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        print(1)
        print(self.suggested_sample)
        self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample)

        for j in range(self.objective.output_dim):
            print(self.Y_new[j])
            self.Y[j] = np.vstack((self.Y[j],self.Y_new[j]))


    def compute_current_best(self):
        current_acqX = self.acquisition.current_compute_acq()
        return current_acqX

    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        return np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))


    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None, re_use=False):

        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """
        ## --- Update the context if any

        self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)
        print("compute next evaluation")
        if self.sample_from_acq:
            print("suggest next location given THOMPSON SAMPLING")
            candidate_points= initial_design('latin', self.space, 2000)
            aux_var = self.acquisition._compute_acq(candidate_points)

        else:

            aux_var = self.evaluator.compute_batch(duplicate_manager=None, re_use=re_use, constrained=False)

        return self.space.zip_inputs(aux_var[0])

    def _update_model(self):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        if (self.num_acquisitions%self.model_update_interval)==0:

            ### --- input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)
            Y_inmodel = list(self.Y)
            self.model.updateModel(X_inmodel, Y_inmodel)

        ### --- Save parameters of the model
        #self._save_model_parameter_values()


    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def func_val(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        Y,_ = self.objective.evaluate(x, true_val=True)
        C,_ = self.constraint.evaluate(x, true_val=True)
        Y = np.array(Y).reshape(-1)
        out = Y.reshape(-1)* np.product(np.concatenate(C, axis=1) < 0, axis=1).reshape(-1)
        out = np.array(out).reshape(-1)
        return -out