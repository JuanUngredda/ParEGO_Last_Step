import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import GPyOpt
import pandas as pd
import os
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from scipy.optimize import minimize

class Last_Step():
    def __init__(self, model_f, model_c, true_f, true_c, n_f, n_c, acquisition_optimiser,acquisition_f,space, seed=None, prior_gen = None,path=None):
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


        self.n_last_steps = 2
        self.path = path

        self.method = "parEGO"

        self.space = space
        self.true_best_x, self.true_best_val = self.acq_opt.optimize_inner_func(f=self.top_true_utility)

        # X_plot = GPyOpt.experiment_design.initial_design('latin', space, 10000)
        # fig, axs = plt.subplots(2, 2)
        #
        #
        #
        # BEST_mu,_ = self.objective.evaluate(self.true_best_x)
        # Y_recommended, cost_new = self.objective.evaluate(X_plot)
        # Y_recommended = -np.concatenate(Y_recommended, axis=1)
        #
        # if self.constraint is not None:
        #     C_recommended, C_cost_new = self.constraint.evaluate(X_plot)
        #     print("C_recommended",C_recommended)
        #     C_recommended = np.concatenate(C_recommended, axis=1)
        #     feasable_samples = np.product(C_recommended < 0, axis=1, dtype=bool)
        #     print("sum feas", np.sum(feasable_samples ))
        #     Feas_muX = Y_recommended[feasable_samples ]
        # else:
        #     Feas_muX = Y_recommended
        #
        # print("Feas_muX ", Feas_muX)
        # uval = self.chevicheff_scalarisation(Feas_muX)
        #
        # # Pdom = self.probability_domination(X_plot[feasable_mu_index])
        # # print("max", np.max(Pdom), "min", np.min(Pdom))
        # print("BEST_mu",BEST_mu, "self.true_best_val",self.true_best_val)
        # print("weights", self.weight, "uval_discre", np.min(uval))
        # axs[0,0].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(uval).reshape(-1))
        # # axs[0, 0].scatter(X_plot[:, 0], X_plot[:, 1], c=np.array(uval).reshape(-1))
        # # axs[0, 0].scatter(-BEST_mu[ 0], -BEST_mu[1], c="red")
        #
        # # axs[0, 0].plot(Feas_muX[:, 0], Feas_muX[:, 0] * (self.weight[:, 0] / self.weight[:, 1]))
        # plt.show()


    def top_true_utility(self,X):
        X = np.atleast_2d(X)
        print("X",X)
        if self.constraint is not None:
            Y_recommended, cost_new = self.objective.evaluate(X)
            C_recommended, C_cost_new = self.constraint.evaluate(X)

            Y_recommended = -np.concatenate(Y_recommended, axis=1)
            C_recommended = np.concatenate(C_recommended, axis=1)

            feasable_samples = np.product(C_recommended < 0, axis=1, dtype=bool)
            uval = self.chevicheff_scalarisation(Y_recommended)
            uval= uval.reshape(-1)
            uval[np.logical_not(feasable_samples)] = 0
            out = uval
            #out = uval.reshape(-1) * feasable_samples.reshape(-1)
        else:
            Y_recommended, cost_new = self.objective.evaluate(X)

            Y_recommended = -np.concatenate(Y_recommended, axis=1)

            uval = self.chevicheff_scalarisation(Y_recommended)
            out = uval.reshape(-1) #* feasable_samples.reshape(-1)
        print(" np.array(out).reshape(-1)", np.array(out).reshape(-1), type(out))
        return np.array(out).reshape(-1)

    def utility(self, objectives):
        y = np.abs(objectives)
        w = self.weight
        scaled_vectors = np.multiply(w, y)
        return np.sum(scaled_vectors, axis=1)

    def prior_sample_generator(self, n_samples=1, seed=None):
        if seed is None:
            samples = np.random.dirichlet(np.ones((self.n_f,)), n_samples)
        else:
            random_state = np.random.RandomState(seed)
            samples = random_state.dirichlet(np.ones((self.n_f,)), n_samples)
        return samples

    def prob_dominance_method(self):
        # Get point from decision maker
        self.ref_point, inner_opt_val = self.acq_opt.optimize_inner_func(f=self.inner_opt)
        #

        # Get point from simulator with greater prob of domination
        # self.probability_domination(simulation_points)
        recommended_x, recommended_x_val = self.acq_opt.optimize_inner_func(f=self.probability_domination,
                                                                            include_point=self.ref_point)
        print("recommended_x, recommended_x_val", recommended_x, recommended_x_val)
        # mu_x_sim = model_f.posterior_mean(dom_opt_x)
        # print("self.ref_point",self.ref_point)
        # print("prob dom ref point", self.probability_domination(self.ref_point))

        plot = False
        if plot == True:

            space = self.space
            X_plot = GPyOpt.experiment_design.initial_design('latin', space, 1000)
            fig, axs = plt.subplots(2, 2)
            muX_inner = self.model.posterior_mean(X_plot)
            muX_inner = np.vstack(muX_inner).T

            Fz = self.probability_feasibility_multi_gp(X_plot)
            feasable_mu_index = np.array(Fz > 0.51, dtype=bool).reshape(-1)
            Feas_muX = muX_inner[feasable_mu_index]
            uval = self.utility(Feas_muX)

            Pdom = self.probability_domination(X_plot[feasable_mu_index])
            print("max", np.max(Pdom), "min", np.min(Pdom))
            axs[0, 0].set_title("utility_plot")
            axs[0, 0].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(uval).reshape(-1))

            mu_x_opt = self.model.posterior_mean(self.ref_point)
            axs[0, 0].scatter(mu_x_opt[0], mu_x_opt[1], color="red")

            mu_recommended = self.model.posterior_mean(recommended_x)
            axs[0, 0].scatter(mu_recommended[0], mu_recommended[1], color="black")

            axs[0, 1].set_title("Pdom_plot")
            axs[0, 1].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(Pdom).reshape(-1))

            plt.show()

        # print("recommended x", recommended_x)

        self.store_results(recommended_x)
        return recommended_x, 0

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

    def hypervolume_method(self):
        self.dm_best_x, inner_opt_val = self.acq_opt.optimize_inner_func(f=self.chevicheff_weight_matching)

        feasable_samples , feasable_Y = self.filter_Y_samples()
        print("feasable_samples , feasable_Y ",feasable_samples , feasable_Y )
        reference_point_hypervolume = self.recompute_reference_pt_hypervolume(feasable_samples, feasable_Y)

        if self.acquisition_f.constraints_flag:
            recommended_x, predicted_f = self.acquisition_f._compute_acq_constraints(ref_point=reference_point_hypervolume, starting_point=self.dm_best_x)
        else:
            wrapped_comput_acq = self._compute_acq_wrapper(ref_point=reference_point_hypervolume, starting_point=self.dm_best_x)
            recommended_x, predicted_f  = self.acq_opt.optimize_inner_func(f=wrapped_comput_acq)

        print("recommended_x",recommended_x, " predicted_f", predicted_f)
        print("reference_point_hypervolume",reference_point_hypervolume)
        #raise
        #include hypervolume   computation
        plot = False
        if plot == True:
            space = self.space
            X_plot = GPyOpt.experiment_design.initial_design('latin', space, 1000)
            fig, axs = plt.subplots(2, 2)
            muX_inner = self.model.posterior_mean(X_plot)
            muX_inner = np.vstack(muX_inner).T
            print("muX_inner ",muX_inner )

            if self.constraint is not None:
                Fz = self.probability_feasibility_multi_gp(X_plot)
                feasable_mu_index = np.array(Fz > 0.51, dtype=bool).reshape(-1)
                Feas_muX = -muX_inner[feasable_mu_index]
            else:
                Feas_muX = -muX_inner

            print("Feas_muX ",Feas_muX )
            uval = self.chevicheff_scalarisation(Feas_muX)

            # Pdom = self.probability_domination(X_plot[feasable_mu_index])
            # print("max", np.max(Pdom), "min", np.min(Pdom))
            axs[0, 0].set_title("utility_plot")
            axs[0, 0].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(uval).reshape(-1))

            mu_x_opt = self.model.posterior_mean(self.dm_best_x)
            mu_x_sampled = self.model.posterior_mean(self.best_sampled_X)
            print("mu_x_sampled ",mu_x_sampled ,"utility best sampled ", self.chevicheff_scalarisation(np.array(mu_x_sampled).reshape(-1)))
            axs[0, 0].scatter(-mu_x_opt[0], -mu_x_opt[1], color="red")
            axs[0,0].scatter(-self.model.get_Y_values()[0],-self.model.get_Y_values()[1], color="magenta" )
            axs[0,0].scatter(-mu_x_sampled[0], -mu_x_sampled[1], color="green")


            # axs[0,0].plot(Feas_muX[:,0],  Feas_muX[:,0] * (self.weight[:,0]/self.weight[:,1]))
            # axs[0,0].set_xlim([-150, 0])
            # axs[0, 0].set_ylim([-70, -30])
            mu_recommended = -self.model.posterior_mean(recommended_x)
            # print("mu_recommended",mu_recommended, "predicted_f",predicted_f)
            axs[0, 0].scatter(mu_recommended[0], mu_recommended[1], color="orange")

            # axs[0, 1].set_title("Pdom_plot")
            # axs[0, 1].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(Pdom).reshape(-1))

            plt.show()


        self.store_results(recommended_x)
        return recommended_x, 0

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

    def chevicheff_scalarisation(self, f):
        y = f
        w = self.weight
        if len(f.shape) == 3:
            w = np.repeat(w[:, :, np.newaxis], f.shape[2], axis=2)
            print("w", w.shape)
            scaled_vectors = w * y  # np.multiply(w, y)
            print("scaled_vectors", scaled_vectors)
            utility = np.max(scaled_vectors, axis=1)
            print("utility", utility)
            return utility
        else:
            scaled_vectors = np.multiply(w, y)
            return np.max(scaled_vectors, axis=1)


    def compute_batch(self, **kwargs):

        if self.method == "pdominance":
            recommended_x, _ = self.prob_dominance_method()
        elif self.method == "hypervolume":
            recommended_x, _ = self.hypervolume_method()

        elif self.method =="parEGO":
            recommended_x, _ = self.parEGO_method()

        elif self.method =="MC_utility_EGO":
            recommended_x, _ = self.MC_EI_true_utility()
        else:
            print("La cagaste, mano")
            raise
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

    def MC_EI_true_utility(self):
        X0 = self.model.get_X_values()
        Y0 = self.model.get_Y_values()
        if self.constraint is None:
            C0 = None
        else:
            C0 = self.model_c.get_Y_values()
        feasable_samples, feasable_Y = self.filter_Y_samples()
        if np.sum(feasable_samples) > 0:
            utility_sampled_designs = self.chevicheff_scalarisation(feasable_Y)
            feasable_sampled_X = self.model.get_X_values()[feasable_samples]
            best_sampled_X = feasable_sampled_X[np.argmin(utility_sampled_designs)]
            self.best_sampled_X = best_sampled_X
            self.base_utility = np.min(utility_sampled_designs)
        else:
            best_sampled_X = np.array([0, 0])
            self.best_sampled_X = best_sampled_X
            self.base_utility = 0.0

        improvement_setp_x, _ = self.acq_opt.optimize_inner_func(f=self.MC_expected_improvement_constrained)
        X_train, Y_train, C_train = self.evaluate_objective(suggested_sample=improvement_setp_x)
        self._update_model(X=X_train, Y=Y_train,C=C_train, model=self.model, model_c = self.model_c)
        recommended_x, _ = self.acq_opt.optimize_inner_func(f=self.chevicheff_weight_matching)

        print("self.base_utility", self.base_utility, "improvement base", _)
        plot = True
        if plot == True:
            space = self.space
            X_plot = GPyOpt.experiment_design.initial_design('latin', space, 5000)
            fig, axs = plt.subplots(1,2)
            muX_inner = self.model.posterior_mean(X_plot)
            muX_inner = np.vstack(muX_inner).T
            print("muX_inner ", muX_inner)

            if self.constraint is not None:
                Fz = self.probability_feasibility_multi_gp(X_plot)
                feasable_mu_index = np.array(Fz > 0.51, dtype=bool).reshape(-1)
                Feas_muX = -muX_inner[feasable_mu_index]
            else:
                Feas_muX = -muX_inner
                feasable_mu_index = np.repeat(True, Feas_muX.shape[0])

            print("Feas_muX ", Feas_muX)
            uval = self.chevicheff_scalarisation(Feas_muX)

            # Pdom = self.probability_domination(X_plot[feasable_mu_index])
            # print("max", np.max(Pdom), "min", np.min(Pdom))
            # axs[0, 0].set_title("utility_plot")
            # axs[0, 0].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(uval).reshape(-1))
            #
            # mu_x_sampled = self.model.posterior_mean(self.best_sampled_X)
            # print("mu_x_sampled ", mu_x_sampled, "utility best sampled ",
            #       self.chevicheff_scalarisation(np.array(mu_x_sampled).reshape(-1)))
            #
            # axs[0, 0].scatter(feasable_Y[:, 0], feasable_Y[:, 1], color="magenta")
            # axs[0, 0].scatter(-mu_x_sampled[0], -mu_x_sampled[1], color="green")

            # axs[0,0].plot(Feas_muX[:,0],  Feas_muX[:,0] * (self.weight[:,0]/self.weight[:,1]))
            # axs[0,0].set_xlim([-150, 0])
            # axs[0, 0].set_ylim([-70, -30])
            mu_recommended = -self.model.posterior_mean(recommended_x)
            # print("mu_recommended",mu_recommended, "predicted_f",predicted_f)
            axs[0].scatter(mu_recommended[0], mu_recommended[1], color="orange")

            #acq = self.MC_expected_improvement_constrained(X_plot[feasable_mu_index])
            # axs[0].set_title("utility_plot")
            axs[0].scatter(Feas_muX[:, 0], Feas_muX[:, 1]), #c=np.array(acq).reshape(-1))

            mu_x_sampled = self.model.posterior_mean(self.best_sampled_X)
            print("mu_x_sampled ", mu_x_sampled, "utility best sampled ",
                  self.chevicheff_scalarisation(np.array(mu_x_sampled).reshape(-1)))

            axs[0].scatter(feasable_Y[:, 0], feasable_Y[:, 1], color="black", label="sampled")
            # axs[0].scatter(-mu_x_sampled[0], -mu_x_sampled[1], color="green")

            # axs[0,0].plot(Feas_muX[:,0],  Feas_muX[:,0] * (self.weight[:,0]/self.weight[:,1]))
            # axs[0,0].set_xlim([-150, 0])
            # axs[0, 0].set_ylim([-70, -30])
            mu_recommended = -self.model.posterior_mean(recommended_x)
            # print("mu_recommended",mu_recommended, "predicted_f",predicted_f)
            axs[0].scatter(mu_recommended[0], mu_recommended[1], color="red", label="DM")
            axs[0].set_ylabel("objective 2")
            axs[0].set_xlabel("objective 1")
            # axs[0].set_xlim(-100,-35)
            axs[0].legend()
            # axs[0, 1].set_title("Pdom_plot")
            # axs[0, 1].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(Pdom).reshape(-1))

            plt.show()

        self._update_model(X=X0, Y=Y0, C = C0, model=self.model, model_c = self.model_c)
        self.store_results(recommended_x)
        return recommended_x, 0


    def MC_expected_improvement_constrained(self, X, offset=1e-4):
        X = np.atleast_2d(X)
        N_samples = X.shape[0]
        MC_sims = 1000

        mu = -self.model.posterior_mean(X).T  # turn problem to maximisation
        var = self.model.posterior_variance(X, noise=False)
        sigma = np.sqrt(var).T
        print("mu", mu.shape, "sigma", sigma.shape)
        output_dim = mu.shape[1]

        sigma = np.repeat(sigma[:, :, np.newaxis], MC_sims, axis=2)
        mu = np.repeat(mu[:, :, np.newaxis], MC_sims, axis=2)
        Z = np.random.normal(0, 1, (N_samples, output_dim, MC_sims))
        print("mu", mu.shape, "sigma", sigma.shape, "Z", Z.shape)
        yn = mu + sigma * Z

        print("yn", yn.shape)
        un = -self.chevicheff_scalarisation(yn)
        print("un", un.shape)
        mu_sample_opt = -self.base_utility  # max value
        imp = un - mu_sample_opt
        imp[imp<0] = 0
        ei = np.mean(imp, axis=1).reshape(-1)
        print("Expected_imp", ei, "max", np.max(ei), "min", np.min(ei))
        #raise
        if self.constraint is not None:
            pf = self.probability_feasibility_multi_gp(x=X).reshape(-1, 1)
            #pf[pf < 0.51] = 0
            print("ei", ei, "pf", pf, "ei *pf ", ei * pf)
            pf = np.array(pf).reshape(-1)
            return -np.array(ei * pf).reshape(-1)
        else:
            print("ei", ei)
            return -np.array(ei).reshape(-1)

    def parEGO_method(self):

        #training GP
        X_train = self.model.get_X_values()
        for it in range(self.n_last_steps):
            Y_train, cost_new = self.objective.evaluate(X_train)
            # print("Y_train",Y_train)
            Y_train = -np.concatenate(Y_train, axis=1)
            U_train = self.chevicheff_scalarisation(Y_train)
            U_train = np.log([U_train.reshape((len(U_train),1))])


            self.model_U = multi_outputGP(output_dim=1, noise_var=[1e-6] , exact_feval=[True] )
            print("model_U,", self.model_U.kernel)
            self._update_model(X_train, U_train, model=self.model_U)

            #finding best sampled point
            feasable_samples, feasable_Y = self.filter_Y_samples()
            if np.sum(feasable_samples)>0:
                # print("feasable_Y",feasable_Y)
                utility_sampled_designs = self.chevicheff_scalarisation(feasable_Y)
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

            recommended_x, _ = self.acq_opt.optimize_inner_func(f=self.expected_improvement_constrained, True_GP=self.model_U,include_point=True)
            # print("self.base_utility",self.base_utility ,"improvement base", _)
            plot = False
            if plot == True:
                space = self.space
                X_plot = GPyOpt.experiment_design.initial_design('latin', space, 5000)
                fig, axs = plt.subplots(3, 2)
                muX_inner, cost = self.objective.evaluate(X_plot)
                # print("muX_inner ", muX_inner)
                muX_inner = np.concatenate(muX_inner,axis=-1)

                if self.constraint is not None:
                    Fz = self.probability_feasibility_multi_gp(X_plot)
                    feasable_mu_index = np.array(Fz > 0.51, dtype=bool).reshape(-1)
                    Feas_muX = -muX_inner[feasable_mu_index]
                else:
                    Feas_muX = -muX_inner
                    feasable_mu_index = np.repeat(True, Feas_muX.shape[0])

                uval = self.model_U.posterior_mean(X_plot)
                uval_var = self.model_U.posterior_variance(X_plot, noise=False)
                print("self.best_sampled_X", self.best_sampled_X)
                mu_x_sampled, cost = self.objective.evaluate(np.atleast_2d(self.best_sampled_X))
                print(" mu_x_sampled,", mu_x_sampled)
                mu_x_sampled = np.concatenate(mu_x_sampled,axis=-1)

                axs[0, 0].set_title("utility_plot GP")
                axs[0, 0].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(uval).reshape(-1))
                axs[0, 0].scatter(feasable_Y[:,0], feasable_Y[:,1], color="magenta")
                axs[0, 0].scatter(-mu_x_sampled[:, 0], -mu_x_sampled[:, 1], color="red", marker="x")

                print("recommended_x",recommended_x, "self.objective.evaluate(recommended_x)",self.objective.evaluate(recommended_x))
                print("log evaluation recommended x", self.expected_improvement_constrained(recommended_x, verbose=True))
                mu_recommended,dummy_C = self.objective.evaluate(recommended_x)
                mu_recommended = -np.concatenate(mu_recommended,axis=-1)
                print("mu_recommended",mu_recommended)

                acq = self.expected_improvement_constrained(X_plot)
                axs[1, 0].set_title("utility_plot")
                axs[1, 0].scatter(Feas_muX[:, 0], Feas_muX[:, 1], c=np.array(acq).reshape(-1))
                axs[1, 0].scatter(feasable_Y[:,0], feasable_Y[:,1], color="magenta")
                axs[1, 0].scatter(mu_recommended[:, 0], mu_recommended[:, 1], color="red")


                utility_underlying_func = self.chevicheff_scalarisation(-muX_inner)

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
        mu = -self.model_U.posterior_mean(X) #turn problem to maximisation
        var = self.model_U.posterior_variance(X, noise=False)

        sigma = np.sqrt(var).reshape(-1, 1)
        mu = mu.reshape(-1,1)

        mu_sample_opt = -self.base_utility # max value


        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt #+ offset
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
            # print("ei", ei)

            if verbose:
                print("ei", ei, "mu", mu, "sigma", sigma)
            return -np.array(ei).reshape(-1)

    def store_results(self, recommended_x):

        feasable_samples, feasable_Y = self.filter_Y_samples()
        if self.constraint is not None:
            Y_recommended, cost_new = self.objective.evaluate(recommended_x)
            C_recommended, C_cost_new = self.constraint.evaluate(recommended_x)

            Y_recommended = -np.concatenate(Y_recommended, axis=1)
            C_recommended = np.concatenate(C_recommended, axis=1)

            feasable_samples = np.product(C_recommended < 0, axis=1)
            uval = self.chevicheff_scalarisation(Y_recommended)
            out = -uval.reshape(-1) * feasable_samples.reshape(-1)

        else:
            Y_recommended, cost_new = self.objective.evaluate(recommended_x)
            Y_recommended = -np.concatenate(Y_recommended, axis=1)
            uval = self.chevicheff_scalarisation(Y_recommended)
            out = -uval.reshape(-1)


        self.chevicheff_scalarisation(np.array(self.model.posterior_mean(recommended_x)).reshape(-1))
        # print("recommended_x)",recommended_x,"metamodel Y",self.model.posterior_mean(recommended_x).reshape(-1),"utility simulated",self.chevicheff_scalarisation(np.array(self.model.posterior_mean(recommended_x)).reshape(-1)),"Y_recommended",Y_recommended,"feasable_samples.reshape(-1)",feasable_samples.reshape(-1),"uval",uval)


        if len(feasable_Y )>0:
            uval_sampled = -np.min(self.chevicheff_scalarisation(feasable_Y))
            print("sampled utilities", self.chevicheff_scalarisation(feasable_Y))
            print("feasable_Y",feasable_Y, "utility sampled", uval_sampled)
            print("Y_recommended ",Y_recommended , "out", out)

        else:
            uval_sampled = 0

        N_entries = len(np.concatenate((self.data["Utility"],np.array(out).reshape(-1))).reshape(-1))
        self.data["Utility"] = np.concatenate((self.data["Utility"],np.array(out).reshape(-1)))
        self.data["Utility_sampled"] = np.concatenate((self.data["Utility_sampled"], np.array(uval_sampled).reshape(-1)))
        self.data["Best_Utility"] = np.repeat(-self.true_best_val,N_entries)


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



