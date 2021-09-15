import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import GPyOpt
import pandas as pd
import os
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from scipy.optimize import minimize
from pygmo import *
from pyDOE import *

class AcquisitionFunctionandDecisionMakerInteraction():
    def __init__(self, model,
                 true_f,
                 acquisition_optimiser,
                 acquisition_f,
                 space,
                 seed=None,
                 NLastSteps=0,
                 InteractionClass =None,
                 Inference_Object=None,
                 path=None):

        self.NLastSteps = NLastSteps
        self.model = model
        self.objective = true_f
        self.acq_opt = acquisition_optimiser
        self.n_f = model.output_dim
        self.seed=seed
        self.acquisition_f = acquisition_f
        self.data = {}
        self.data["Utility"] = np.array([])
        self.data["Utility_sampled"] = np.array([])
        self.data["Best_Utility"] = np.array([])
        self.space = space
        self.InteractionClass =InteractionClass
        self.Inference_Object = Inference_Object
        self.path =path

    def get_optimiser(self):
        return self.acq_opt.optimize_inner_func

    def get_true_parameters(self):
        return self.InteractionClass.get_true_parameters()

    def get_algorithm_utility(self):
        return self.Inference_Object.get_utility_function()

    def get_true_utility(self):
        return self.InteractionClass.get_true_utility()

    def include_posterior_samples_in_acq(self, posterior_samples):
        self.acquisition_f.include_fantasised_posterior_samples(posterior_samples)

    def get_data(self):
        return self.InteractionClass.get_

    def _update_model(self, X, Y, model=None):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """

        ### --- input that goes into the model (is unziped in case there are categorical variables)
        X_inmodel = self.space.unzip_inputs(X)
        model.updateModel(X_inmodel, Y)


    def get_DecisionMaker_data(self):
        return self.InteractionClass.ShowParetoFronttotheDecisionMaker()

    def posterior_sample_generation(self,solution_picked_dm_index, PF):

        self.Inference_Object.update_sampled_data(PF, solution_picked_dm_index)
        posterior_samples = self.Inference_Object.posterior_sampler(n_samples=50)
        return posterior_samples

    def acquisition_function(self, X):
        return -self.acquisition_f._compute_acq(X)

    def nLastStepsCaller(self):

        #get initial data
        X_train = self.model.get_X_values()
        Y_train = self.model.get_Y_values()
        original_X_train  = self.model.get_X_values()
        original_Y_train = self.model.get_Y_values()

        Pareto_fronts = []
        picked_solution_indeces = []

        for it in range(self.NLastSteps):

            # Sampling preferred solution from DM
            solution_picked_dm_index, solution_x_picked_dm, PF, _ = self.get_DecisionMaker_data()

            Pareto_fronts.append(PF)
            picked_solution_indeces.append(solution_picked_dm_index)
            # Inference on pareto front and getting posterior samples
            posterior_samples = self.posterior_sample_generation(picked_solution_indeces,
                                                                 Pareto_fronts)

            #include fantasised posterior samples
            self.acquisition_f.include_fantasised_posterior_samples(posterior_samples)

            #get recommended design
            recommended_x, _ = self.acq_opt.optimize_inner_func(f=self.acquisition_function,
                                                                include_point=None)

            plot = False
            if plot == True:

                fig, axs = plt.subplots(2, 2)

                #get (X,Y) data from the GP
                space = self.space
                weight = self.get_true_parameters()
                algorithm_utility = self.get_algorithm_utility()

                X_plot = GPyOpt.experiment_design.initial_design('latin', space, 5000)
                muX_inner = self.model.posterior_mean(X_plot) # self.objective.evaluate(X_plot)
                muX_inner = np.vstack(muX_inner).T


                Feas_muX = muX_inner #convert the problem into minimisation
                Y_train = np.hstack(Y_train)
                YSampledUtility = algorithm_utility(Y_train,
                                      weights=weight[1],
                                      parameters=weight[0])

                BestSampledUtility = X_train[np.argmax(YSampledUtility)]

                Ytrue, _ = self.objective.evaluate(X_plot)
                Ytrue = np.hstack(Ytrue)
                YtrueUtility =[]
                for ytrue in Ytrue:
                    YtrueUtility.append(algorithm_utility(ytrue,
                                                          weights=weight[1],
                                                          parameters=weight[0]))

                X_true_best = X_plot[np.argmax(YtrueUtility)]
                Y_true_best = Ytrue[np.argmax(YtrueUtility)]

                axs[0, 0].set_title("X_plot "+str(it))
                axs[0, 0].scatter(X_plot[:, 0], X_plot[:, 1], c=np.array(YtrueUtility).reshape(-1))
                axs[0, 0].scatter(X_train[:,0], X_train[:,1], color="magenta")
                axs[0, 0].scatter(recommended_x[:, 0], recommended_x[:, 1], color="red", marker="x")
                axs[0, 0].scatter(BestSampledUtility[0],BestSampledUtility[1], color="black", marker="x")
                axs[0, 0].scatter(X_true_best[0],X_true_best[1], color="black")
                axs[0, 0].set_xlim([-1,1])
                axs[0, 0].set_ylim([-1,1])

                #compute objective value of recommended solution by expected improvement
                # mu_recommended,dummy_C = self.objective.evaluate(recommended_x)
                mu_recommended = self.model.posterior_mean(recommended_x) # self.objective.evaluate(X_plot)
                mu_recommended = np.vstack(mu_recommended).T
                # mu_recommended = -np.concatenate(mu_recommended,axis=-1)
                print("mu_recommended",mu_recommended)

                acq = self.acquisition_f._compute_acq(X_plot)
                axs[1, 0].set_title("utility_plot")
                axs[1, 0].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(acq).reshape(-1))
                axs[1, 0].scatter(Y_train[:,0], Y_train[:,1], color="magenta")
                axs[1, 0].scatter(mu_recommended[:, 0], mu_recommended[:, 1], marker="x", color="red")
                axs[1, 0].scatter(Y_true_best[0], Y_true_best[1], color="red")


                axs[1, 1].set_title("utility_plot")
                axs[1, 1].scatter(Ytrue[:, 0], Ytrue[:, 1])
                axs[1, 1].scatter(Y_true_best[0], Y_true_best[1], color="red")

                posterior_parameteres = posterior_samples[0]
                axs[0, 1].set_title("posterior samples")
                axs[0, 1].hist(posterior_parameteres[0][:,0], density=True)

                plt.show()

            X_train = np.concatenate((X_train, recommended_x))
            #train model_output

            Y_train, cost_new = self.objective.evaluate(X_train)
            self._update_model(X_train, Y_train, model=self.model)

        self.store_results(np.atleast_2d(recommended_x))
        #restarting variables previous to Decision maker elicitation.
        self.acquisition_f.include_fantasised_posterior_samples(None)
        self._update_model(original_X_train, original_Y_train, model=self.model)

        return 0


    def store_results(self, recommended_x):

        true_underlying_utility = self.get_true_utility()
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
        self.data["Utility_sampled"] = np.concatenate((self.data["Utility_sampled"], np.array(uval_sampled).reshape(-1)))

        true_best_x, true_best_val = self.compute_underlying_best()
        self.data["Best_Utility"] = np.repeat(true_best_val , N_entries)

        if self.path is not None:
            gen_file = pd.DataFrame.from_dict(self.data)
            results_folder = "Utility"

            path = self.path +"/" + results_folder + '/it_' + str(self.seed) + '.csv'
            if os.path.isdir(self.path +"/" + results_folder  ) == False:
                os.makedirs(self.path +"/" + results_folder  )

            gen_file.to_csv(path_or_buf=path)

            extra_data ={"parameters":true_parameters, "numberofsteps": self.NLastSteps}
            extra_gen_file = pd.DataFrame.from_dict(extra_data)
            extra_path = self.path + "/" + results_folder + '/parameters_'+str(self.seed)+'.csv'
            extra_gen_file.to_csv(path_or_buf=extra_path)


    def compute_underlying_best(self):

        true_underlying_utility = self.get_true_utility()
        weight = self.get_true_parameters()
        def top_true_utility(X):
            X = np.atleast_2d(X)
            Y_recommended, cost_new = self.objective.evaluate(X)
            Y_recommended = -np.concatenate(Y_recommended, axis=1)
            uval = true_underlying_utility(y = Y_recommended,
                                           weights= weight[1],
                                           parameters= weight[0])

            return np.array(uval).reshape(-1)

        true_best_x, true_best_val = self.acq_opt.optimize_inner_func(f=top_true_utility)

        return true_best_x, true_best_val

