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

class Last_Step_step_util():
    def __init__(self, model_f, model_c, true_f, true_c, n_f, n_c, acquisition_optimiser,acquisition_f,space, B=1, seed=None, prior_gen = None,path=None):
        self.model = model_f
        self.model_c = model_c
        self.objective = true_f
        self.constraint = true_c
        self.acq_opt = acquisition_optimiser
        self.n_f = n_f
        self.n_c = n_c
        self.seed=seed
        self.weight = prior_gen(n_samples=1, seed=seed) #np.array([[0.5,0.5]])#
        print("self.weight",self.weight)
        self.acquisition_f = acquisition_f
        self.data = {}
        self.data["Utility"] = np.array([])
        self.data["Utility_sampled"] = np.array([])
        self.data["Best_Utility"] = np.array([])
        for w in range(self.weight.shape[1]):
            self.data["w" + str(w)] = np.array([])


        self.n_last_steps = B
        self.path = path

        self.method = "parEGO"

        self.space = space

        self.DM_utility = self.linear_utility
        self.Alg_utility = self.chevicheff_scalarisation
        self.true_best_x, self.true_best_val = self.acq_opt.optimize_inner_func(f=self.top_true_utility)

        print("self.true_best_val",self.true_best_val)


    def wrapper(self, f):
        def changed_func(x):
            output,c = f(x)
            output = np.vstack(output).T
            # print("output", output)
            return output
        return changed_func

    def top_true_utility(self,X):
        X = np.atleast_2d(X)
        # print("X",X)
        if self.constraint is not None:
            Y_recommended, cost_new = self.objective.evaluate(X)
            C_recommended, C_cost_new = self.constraint.evaluate(X)

            Y_recommended = -np.concatenate(Y_recommended, axis=1)
            C_recommended = np.concatenate(C_recommended, axis=1)

            feasable_samples = np.product(C_recommended < 0, axis=1, dtype=bool)
            uval = self.DM_utility(Y_recommended, w=self.weight)
            uval= uval.reshape(-1)
            uval[np.logical_not(feasable_samples)] = 0
            out = uval
            #out = uval.reshape(-1) * feasable_samples.reshape(-1)
        else:
            Y_recommended, cost_new = self.objective.evaluate(X)
            Y_recommended = -np.concatenate(Y_recommended, axis=1)
            uval = self.DM_utility(Y_recommended, w=self.weight)
            out = uval.reshape(-1) #* feasable_samples.reshape(-1)
        # print(" np.array(out).reshape(-1)", np.array(out).reshape(-1), type(out))
        return np.array(out).reshape(-1)


    def linear_utility(self, f, w):

        y = f
        w = np.atleast_2d(w)
        scaled_vectors = np.multiply(w, y)
        u = np.sum(scaled_vectors, axis=1).reshape(-1)
        # print("f", f, "w", w, "u", u)
        return u


    def chevicheff_scalarisation(self, f, w):
        y = f
        w = np.atleast_2d(w)
        if len(f.shape) == 3:
            w = np.repeat(w[:, :, np.newaxis], f.shape[2], axis=2)

            scaled_vectors = w * y  # np.multiply(w, y)

            utility = np.max(scaled_vectors, axis=1)

            return utility
        else:
            utility = []
            for weights in w:
                weights = np.atleast_2d(weights)
                scaled_vectors = np.multiply(weights, y)
                utility.append(np.max(scaled_vectors, axis=1))
            return np.array(utility).reshape(-1)

    def prior_sample_generator(self, n_samples=1, seed=None):
        if seed is None:
            samples = np.random.dirichlet(np.ones((self.n_f,)), n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.dirichlet(np.ones((self.n_f,)), n_samples)
        return samples



    def get_ref_point(self, X_best, best_utility):
        mu_Xbest = self.model.posterior_mean(X_best)
        mu_Xbest = -np.vstack(mu_Xbest).T
        w = np.array(self.weight).reshape(-1)
        scaled_vectors = np.multiply(w, mu_Xbest)
        best_f_idx = np.argmax(scaled_vectors)
        recip_w = np.reciprocal(w)
        reference_w = np.array(w * recip_w).reshape(-1)
        reference_vector = np.zeros(len(reference_w))
        for idx in range(len(reference_w)):
            reference_vector[idx] = np.max(mu_Xbest )* (w[best_f_idx]/w[idx])
        return reference_vector

    def _compute_acq_wrapper(self, ref_point, starting_point):
        def inner_function_HVI(X):
                return -self.acquisition_f._compute_acq(X=X, ref_point=ref_point, starting_point=starting_point)
        return inner_function_HVI

    def filter_Y_samples(self):
        sampled_X = self.model.get_X_values()
        sampled_Y,cost = self.objective.evaluate(sampled_X)
        sampled_Y = -np.concatenate(sampled_Y, axis=1)
        if self.constraint is not None:
            C_true = self.model_c.get_Y_values()
            feasable_samples = np.product(np.concatenate(C_true, axis=1) < 0, axis=1)
            feasable_samples = np.array(feasable_samples, dtype=bool)
        else:
            feasable_samples = np.repeat(True, sampled_Y.shape[0])

        feasable_Y = sampled_Y[feasable_samples]
        return feasable_samples , feasable_Y

    def recompute_reference_pt_hypervolume(self, feasable_samples, feasable_Y):
        if np.sum(feasable_samples)>0:

            utility_sampled_designs = self.chevicheff_scalarisation(feasable_Y)
            print("utility_sampled_designs ",utility_sampled_designs )
            feasable_sampled_X = self.model.get_X_values()[feasable_samples]
            best_sampled_X = feasable_sampled_X[np.argmin(utility_sampled_designs )]
            self.best_sampled_X = best_sampled_X
            reference_point_hypervolume = self.get_ref_point(best_sampled_X, best_utility=np.min(utility_sampled_designs))
        else:
            best_sampled_X =np.array([0,0])
            self.best_sampled_X = best_sampled_X
            reference_point_hypervolume = np.array([0,0])

        return reference_point_hypervolume


    def chevicheff_weight_matching(self,X):
        X = np.atleast_2d(X)

        if self.constraint is not None:
            Pfeas = self.probability_feasibility_multi_gp(X)
            mu_x = self.model.posterior_mean(X)
            mu_x = -np.vstack(mu_x).T
            uval = self.chevicheff_scalarisation( f = mu_x)
            out = uval.reshape(-1) * Pfeas.reshape(-1)
        else:
            mu_x = self.model.posterior_mean(X)
            mu_x = -np.vstack(mu_x).T
            uval = self.chevicheff_scalarisation( f = mu_x)
            out = uval.reshape(-1)
        return out

    def compute_batch(self, **kwargs):

        recommended_x, _ = self.parEGO_method()

        return recommended_x, 0

    def evaluate_objective(self, suggested_sample):
        """
        Evaluates the objective
        """
        suggested_sample = np.atleast_2d(suggested_sample)
        X = self.model.get_X_values()
        Y, cost_new = self.objective.evaluate(X)


        Y_new, cost_new = self.objective.evaluate(suggested_sample)
        for j in range(self.objective.output_dim):
            Y[j] = np.vstack((Y[j], Y_new[j]))

        if self.constraint is not None:
            C = self.constraint.evaluate(X)
            C_new, C_cost_new = self.constraint.evaluate(suggested_sample)

            for k in range(self.constraint.output_dim):

                C[k] = np.vstack((C[k], C_new[k]))

            X = np.vstack((X,suggested_sample))
            return X, Y, C
        else:
            X = np.vstack((X,suggested_sample))
            return X, Y , 0

    def _update_model(self, X, Y,C=None, model=None, model_c = None):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """

        ### --- input that goes into the model (is unziped in case there are categorical variables)

        X_inmodel = self.space.unzip_inputs(X)

        # print("X_inmodel, Y_inmodel",X_inmodel, Y_inmodel)
        model.updateModel(X_inmodel, Y)
        if self.constraint is not None:
            if len(C) == 1:
                C_inmodel = [C]
            else:
                C_inmodel = C
            model_c.updateModel(X_inmodel, C_inmodel)


    def Generate_Pareto_Front(self):
        X_train = self.model.get_X_values()

        Y_train, cost_new = self.objective.evaluate(X_train)
        model = multi_outputGP(output_dim=2, noise_var=[1e-6,1e-6] , exact_feval=[True,True] )
        self._update_model(X_train, Y_train, model=model)
        GP_y_predictions  = self.mean_prediction_model(X_train,model)


        bounds = self.space.get_continuous_bounds()
        bounds = self.bounds_format_adapter(bounds)
        udp = GA(f=GP_y_predictions, bounds=bounds, n_obj=len(Y_train))
        pop = population(prob=udp, size=100)
        algo = algorithm(nsga2(gen=300))
        pop = algo.evolve(pop)
        fits, vectors = pop.get_f(), pop.get_x()
        ndf, dl, dc, ndr = fast_non_dominated_sorting(fits)
        result_x = vectors[ndf[0]]
        result_fx = fits[ndf[0]]
        # # X_plot = GPyOpt.experiment_design.initial_design('latin', self.space, 5000)
        # # plot_y = GP_y_predictions(X_plot)
        #
        # # plt.scatter(-plot_y[:,0], -plot_y[:,1], color="red")
        # plt.scatter(result_fx[:, 0], result_fx[:, 1], color="blue")
        #
        # plt.show()

        # data_pf = {}
        # data_sampled = {}
        #
        # # data_pf["PF1"] =np.array(result_fx[:,0]).reshape(-1)
        # # data_pf["PF2"] =np.array(result_fx[:,1]).reshape(-1)
        # Y_train = np.hstack(Y_train)
        # print("Y_train", Y_train)
        # data_sampled["Y1"] = np.array(Y_train[:,0]).reshape(-1)
        # data_sampled["Y2"] = np.array(Y_train[:, 1]).reshape(-1)
        # gen_file = pd.DataFrame.from_dict(data_sampled)
        # path = "/home/juan/Documents/repos_data/ParEGO_Last_STEP/Paper_PLOTS/Problem_Def_PLOT/GP_Y_data.csv"
        # gen_file.to_csv(path_or_buf=path)
        #
        #
        # # gen_file = pd.DataFrame.from_dict(data_pf)
        # # path = "/home/juan/Documents/repos_data/ParEGO_Last_STEP/Paper_PLOTS/Problem_Def_PLOT/GP_PF1.csv"
        # # gen_file.to_csv(path_or_buf=path)
        #
        # raise
        return result_fx, result_x#GP_y_predictions

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



    def parEGO_method(self):

        #training GP
        X_train = self.model.get_X_values()
        PF, PF_Xvals = self.Generate_Pareto_Front() #generate pareto front
        Utility_PF =  self.DM_utility(PF, w=self.weight) #dm picks a solution

        #solution picked by the dm in output space
        self.solution_picked_dm = PF[np.argmin(Utility_PF)] #minimise utility (?)
        self.solution_x_picked_dm = PF_Xvals[np.argmin(Utility_PF)]

        w_random = self.prior_sample_generator(1) #get random

        print("true weight", self.weight, "estimated weight", w_random)
        for it in range(self.n_last_steps):
            print("Last step acquired samples Main Alg:", it)
            Y_train, cost_new = self.objective.evaluate(X_train)
            # print("Y_train",Y_train)

            self.model_outputs = multi_outputGP(output_dim=2, noise_var=[1e-6]*2, exact_feval=[True]*2)
            self._update_model(X_train,Y_train, model=self.model_outputs)

            Y_train = np.concatenate(Y_train, axis=1)
            print("Y_train",Y_train)

            U_train = self.Alg_utility(Y_train, w=w_random)
            U_train = np.array([U_train.reshape((len(U_train),1))])

            self.model_U = multi_outputGP(output_dim=1, noise_var=[1e-6] , exact_feval=[True] )

            self._update_model(X_train, U_train, model=self.model_U)

            #finding best sampled point
            feasable_samples, feasable_Y = self.filter_Y_samples()
            if np.sum(feasable_samples)>0:
                # print("feasable_Y",feasable_Y)
                utility_sampled_designs = self.Alg_utility(feasable_Y, w=w_random)
                # print("U_train",U_train)
                # print("utility_sampled_designs",utility_sampled_designs)

                feasable_sampled_X = self.model.get_X_values()[feasable_samples]
                best_sampled_X = feasable_sampled_X[np.argmin(utility_sampled_designs)]
                self.best_sampled_X = best_sampled_X
                self.base_utility = np.min(utility_sampled_designs)
                print("self.base_utility",self.base_utility)
                # print("U_train",U_train)
            else:
                best_sampled_X = np.array([0, 0])
                self.best_sampled_X = best_sampled_X
                self.base_utility = 0.0

            print("self.solution_x_picked_dm ",self.solution_x_picked_dm )
            recommended_x, _ = self.acq_opt.optimize_inner_func(f=self.expected_improvement_constrained,
                                                                True_GP=self.model_U,
                                                                include_point=np.atleast_2d(self.solution_x_picked_dm ))

            plot = False
            if plot == True:

                fig, axs = plt.subplots(2, 2)

                #get (X,Y) data from the GP
                space = self.space
                X_plot = GPyOpt.experiment_design.initial_design('latin', space, 5000)
                muX_inner = self.model_outputs.posterior_mean(X_plot) # self.objective.evaluate(X_plot)
                muX_inner = np.vstack(muX_inner).T

                #get (X,U) data from the GP
                uval = self.model_U.posterior_mean(X_plot) #for lower utility values, the objective should be better.
                uval_var = self.model_U.posterior_variance(X_plot, noise=False)

                if self.constraint is not None:
                    Fz = self.probability_feasibility_multi_gp(X_plot)
                    feasable_mu_index = np.array(Fz > 0.51, dtype=bool).reshape(-1)
                    Feas_muX = -muX_inner[feasable_mu_index] #get feasible samples and convert to minimisation
                else:
                    Feas_muX = -muX_inner #convert the problem into minimisation
                    feasable_mu_index = np.repeat(True, Feas_muX.shape[0])

                # objective value of the sampled value with minimal utility
                mu_x_sampled, cost = self.objective.evaluate(np.atleast_2d(self.best_sampled_X))
                mu_x_sampled = -np.concatenate(mu_x_sampled,axis=-1)

                axs[0, 0].set_title("utility_plot GP")
                axs[0, 0].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(uval).reshape(-1))
                axs[0, 0].scatter(feasable_Y[:,0], feasable_Y[:,1], color="magenta")
                axs[0, 0].scatter(mu_x_sampled[:, 0], mu_x_sampled[:, 1], color="red", marker="x")
                axs[0, 0].scatter(self.solution_picked_dm[0], self.solution_picked_dm[1], color="black", marker="x")
                axs[0 , 0].set_xlim([0,8])
                axs[0 , 0].set_ylim([0,8])

                #compute objective value of recommended solution by expected improvement
                mu_recommended,dummy_C = self.objective.evaluate(recommended_x)
                mu_recommended = -np.concatenate(mu_recommended,axis=-1)
                print("mu_recommended",mu_recommended)

                acq = self.expected_improvement_constrained(X_plot)

                print("acq", acq, np.min(acq), np.max(acq))
                axs[1, 0].set_title("utility_plot")
                axs[1, 0].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(acq).reshape(-1))
                axs[1, 0].scatter(feasable_Y[:,0], feasable_Y[:,1], color="magenta")
                axs[1, 0].scatter(mu_recommended[:, 0], mu_recommended[:, 1], color="red")

                plt.show()
                raise
                utility_underlying_func = self.Alg_utility(-muX_inner, w=w_random)

                axs[0, 1].set_title("utility_plot true")
                axs[0, 1].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(utility_underlying_func).reshape(-1))
                axs[0, 1].scatter(feasable_Y[:, 0], feasable_Y[:, 1], color="magenta")
                axs[0, 1].scatter(-mu_x_sampled[:, 0], -mu_x_sampled[:, 1], color="red", marker="x")


                axs[1, 1].set_title("utility_plot X space true")
                axs[1, 1].scatter(X_plot[:, 0], X_plot[:, 1], c=np.array(uval_var).reshape(-1))
                axs[1, 1].scatter(X_train[:,0], X_train[:,1], c="magenta")

                axs[2, 1].set_title("utility_plot X space GP")
                axs[2, 1].scatter(X_plot[:, 0], X_plot[:, 1], c=np.array(uval).reshape(-1))
                axs[2, 1].scatter(recommended_x[:, 0], recommended_x[:, 1], color="red")

                print("disc acq", np.min(acq), "true minimum",_ )
                axs[2, 0].set_title("utility_plot X space acq")
                axs[2, 0].scatter(X_plot[:, 0], X_plot[:, 1], c=np.array(acq).reshape(-1))
                axs[2,0].scatter(recommended_x[:,0], recommended_x[:,1], color="red")
                axs[2, 0].scatter(X_train[:, 0], X_train[:, 1], c="magenta")
                axs[2, 0].scatter(self.best_sampled_X[0], self.best_sampled_X[1], color="red", marker="x")

                # axs[0,0].plot(Feas_muX[:,0],  Feas_muX[:,0] * (self.weight[:,0]/self.weight[:,1]))
                # axs[0,0].set_xlim([-150, 0])
                # axs[0, 0].set_ylim([-70, -30])
                # mu_recommended = -self.model.posterior_mean(recommended_x)
                # print("mu_recommended",mu_recommended, "predicted_f",predicted_f)

                # axs[0, 1].set_title("Pdom_plot")
                # axs[0, 1].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(Pdom).reshape(-1))

                plt.show()

            X_train = np.concatenate((X_train,recommended_x))

            self.store_results(np.atleast_2d(recommended_x))

        return recommended_x, 0

    def expected_improvement_constrained(self, X, offset=1e-4, verbose=False):
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
        mu = -self.model_U.posterior_mean(X) #negative values of the utility
        var = self.model_U.posterior_variance(X, noise=False)

        sigma = np.sqrt(var).reshape(-1, 1)
        mu = mu.reshape(-1,1)

        mu_sample_opt = self.base_utility # currently this is min sampled value

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu  #+ offset
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        if self.constraint is not None:
            print("X,self.model_c",X,self.model_c)
            pf = self.probability_feasibility_multi_gp(x =X).reshape(-1,1)
            pf[pf<0.51] = 0
            print("ei", ei, "pf", pf, "ei *pf ",ei *pf )
            raise
            return -np.array(ei *pf ).reshape(-1)
        else:

            mu_output_gps =  -self.model_outputs.posterior_mean(X).T

            ei_vals = []
            for mu_gps in range(len(mu_output_gps)):
                if pareto_dominance(self.solution_picked_dm, mu_output_gps[mu_gps]):
                    ei_vals.append(ei[mu_gps]*0)
                elif pareto_dominance(mu_output_gps[mu_gps], self.solution_picked_dm):
                    ei_vals.append(ei[mu_gps]*1)
                else:
                    ei_vals.append(ei[mu_gps] * (1/3))

            if verbose:
                print("ei", ei, "mu", mu, "sigma", sigma)
            return -np.array(ei_vals).reshape(-1)

    def store_results(self, recommended_x):

        feasable_samples, feasable_Y = self.filter_Y_samples()

        Y_recommended, cost_new = self.objective.evaluate(recommended_x)
        Y_recommended = -np.concatenate(Y_recommended, axis=1)
        uval = self.DM_utility(Y_recommended, w=self.weight)
        out = -uval.reshape(-1)

        uval_sampled = -np.min(self.DM_utility(feasable_Y, w=self.weight))
        print("sampled utilities", self.DM_utility(feasable_Y, w=self.weight))
        print("feasable_Y",feasable_Y, "utility sampled", uval_sampled)
        print("Y_recommended ",Y_recommended , "out", out)

        N_entries = len(np.concatenate((self.data["Utility"], np.array(out).reshape(-1))).reshape(-1))
        self.data["Utility"] = np.concatenate((self.data["Utility"], np.array(out).reshape(-1)))
        self.data["Utility_sampled"] = np.concatenate((self.data["Utility_sampled"], np.array(uval_sampled).reshape(-1)))
        self.data["Best_Utility"] = np.repeat(-self.true_best_val, N_entries)


        for w in range(self.weight.shape[1]):
            self.data["w"+str(w)] = np.concatenate((self.data["w"+str(w)], np.array([self.weight[:,w]]).reshape(-1)))

        if self.path is not None:
            gen_file = pd.DataFrame.from_dict(self.data)
            results_folder = "Utility_Improvement"

            path = self.path +"/" + results_folder + '/it_' + str(self.seed) + '.csv'
            if os.path.isdir(self.path +"/" + results_folder  ) == False:
                os.makedirs(self.path +"/" + results_folder  )

            gen_file.to_csv(path_or_buf=path)


    def inner_opt(self, X):
        X = np.atleast_2d(X)

        mu_x = self.model.posterior_mean(X)
        mu_x = np.vstack(mu_x).T
        Pfeas = self.probability_feasibility_multi_gp(X)
        uval = self.utility(mu_x)
        out = uval.reshape(-1) * Pfeas.reshape(-1)

        return -out

    def probability_feasibility_multi_gp(self,x):
        # print("model",model.output)
        x = np.atleast_2d(x)
        Fz = []
        for m in range(self.model_c.output_dim):
            Fz.append(self.probability_feasibility(x, self.model_c.output[m], l=0))
        Fz = np.product(Fz, axis=0)
        return Fz

    def probability_feasibility(self, x, model, l=0):
        x = np.atleast_2d(x)
        mean = model.posterior_mean(x)
        var = model.posterior_variance(x, noise=False)
        std = np.sqrt(var).reshape(-1, 1)

        mean = mean.reshape(-1, 1)

        norm_dist = norm(mean, std)
        Fz = norm_dist.cdf(l)

        return Fz.reshape(-1, 1)

    def probability_domination(self, X):
        X = np.atleast_2d(X)
        # print("X", X.shape)
        mu = self.model.posterior_mean(X)
        var = self.model.posterior_variance(X, noise=False)
        mu_ref = self.model.posterior_mean(self.ref_point)
        mu_ref = mu_ref.reshape(-1)

        Pdom = np.zeros((X.shape[0], len(mu)))
        for idx in range(len(mu)):
            rv = norm(mu[idx], var[idx])
            pdom =  1 - rv.cdf(mu_ref[idx])
            Pdom[:,idx] = pdom
        Pdom = np.product(Pdom, axis=1)

        Fz = self.probability_feasibility_multi_gp(X)
        Pdom = np.array(Pdom).reshape(-1) * Fz.reshape(-1)
        return -Pdom


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
