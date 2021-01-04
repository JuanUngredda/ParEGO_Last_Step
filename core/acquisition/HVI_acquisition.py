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


class HVI(AcquisitionBase):
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
        super(HVI, self).__init__(model, space, optimizer, model_c = model_c, alpha =alpha, NSGA_based=NSGA_based,cost_withGradients=cost_withGradients)
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
        if ref_point is None:
            self.last_step_flag = False
            self.ref_point = self.ref_point_computation()
            self.alpha = 1.95
        else:
            print(" ref_point used", ref_point)
            self.last_step_flag = True
            self.ref_point = ref_point
            self.starting_point= starting_point
            self.alpha = 0

        HVI = self.compute_HVI_LCB(X)
        return np.array(HVI).reshape(-1)


    def _compute_acq_constraints(self, ref_point=None, starting_point=None):


        if ref_point is None:
            self.last_step_flag = False
            self.ref_point = self.ref_point_computation()
            self.alpha = 1.95
        else:
            print(" ref_point used", ref_point)
            self.last_step_flag = True
            self.ref_point = ref_point
            self.starting_point= starting_point
            self.alpha = 0

        recommended_x , predicted_f  = self.compute_HVI_LCB_constrained()

        print("_compute_acq_constraints", recommended_x , predicted_f )
        return (recommended_x , predicted_f)

    def compute_HVI_LCB_last_step(self, X):
        X = np.atleast_2d(X)
        print("X",X)
        HV0 = 0
        HV_new = np.zeros(X.shape[0])

        for i in range(len(X)):
            x = np.atleast_2d(X[i])
            ## Hypervolume computations
            print("x",x)
            mu = -self.model.posterior_mean(x)
            var = self.model.posterior_variance(x, noise=False)
            y_lcb = mu - self.alpha * np.sqrt(var)
            y_lcb = np.transpose(y_lcb)
            # print("np.array(mu).reshape(-1) ",np.array(y_lcb).reshape(-1)  )
            # print("self.ref_point",self.ref_point)
            # print("np.array(y_lcb).reshape(-1) >  self.ref_point",np.array(y_lcb).reshape(-1) >  self.ref_point)
            # print("np.product(np.array(y_lcb).reshape(-1) >  self.ref_point)",np.product(np.array(y_lcb).reshape(-1) >  self.ref_point))
            if np.sum(np.array(y_lcb).reshape(-1) >  self.ref_point)>0:
                HV_new[i] = -999999
            else:
                hv_new =hypervolume(y_lcb.tolist())
                HV_new[i] = hv_new.compute(ref_point = self.ref_point)
        HVI = HV_new - HV0
        HVI[HVI < 0.0] = 0.0
        return -HVI

    def HV_PF(self,X):

        X = np.atleast_2d(X)
        HV = self.compute_HVI_LCB_last_step(X)
        PF = np.array(self.probability_feasibility_multi_gp_wrapper(model=self.model_c, l=0)(X)).reshape(-1)
        PF[PF<0.51] = 0
        return np.array(HV).reshape(-1) * np.array(PF).reshape(-1)

    def compute_HVI_LCB_constrained(self):

        if self.last_step_flag:
            #f = [self.compute_HVI_LCB_last_step, self.probability_feasibility_multi_gp_wrapper(model=self.model_c, l=0)]
            recommended_x, recommended_f= self.optimizer.optimize_inner_func(f=self.HV_PF, include_point=np.atleast_2d(self.starting_point))
            recommended_f = -recommended_f
            recommended_x = np.atleast_2d(recommended_x)
            return recommended_x, recommended_f
        else:
            f = [self.compute_HVI_LCB, self.probability_feasibility_multi_gp_wrapper(model=self.model_c, l=0)]

            inner_opt_x, inner_opt_val = self.optimizer.optimize_inner_func_constraints(f=f)
            inner_opt_val = -inner_opt_val

            feasable_x = inner_opt_x[inner_opt_val[:,1]>0.51]
            feasable_f = inner_opt_val[inner_opt_val[:,1]>0.51]

            if len(feasable_f[:,1]) ==0:
                recommended_x = inner_opt_x[np.argmax(inner_opt_val[:,1])]
                recommended_f = np.max(inner_opt_val[:,1])
            else:
                recommended_x = feasable_x[np.argmin(feasable_f[:, 1])]
                recommended_f = feasable_f[np.argmin(feasable_f[:, 1])]

            # plt.scatter(inner_opt_val[:,0],inner_opt_val[:,1], color="blue")
            # plt.scatter(recommended_f[0],recommended_f[1], color="red")
            # plt.show()

            # import GPyOpt
            # space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (0.1, 1)},
            #                                    {'name': 'var_2', 'type': 'continuous', 'domain': (0, 5)}])
            # design_plot = initial_design('random', space, 1000)
            #
            # mu = -self.model.posterior_mean(design_plot)
            # mu = np.vstack(mu).T
            # Fz = self.probability_feasibility_multi_gp_wrapper(self.model_c)(design_plot).reshape(-1, 1)
            #
            # feasable_mu_index = np.array(Fz > 0.51, dtype=bool).reshape(-1)
            # Feas_muX = mu[feasable_mu_index]
            #
            # recommended_mu = -self.model.posterior_mean(recommended_x)
            # plt.scatter(Feas_muX[:, 0], Feas_muX[:, 1])  #
            # plt.scatter(recommended_mu[ 0], recommended_mu[1])
            # plt.show()

    #        raise
            recommended_x = np.atleast_2d(recommended_x)
            return recommended_x, recommended_f

    def compute_HVI_LCB(self, X):
        X = np.atleast_2d(X)
        if self.last_step_flag:
            # f = [self.compute_HVI_LCB_last_step, self.probability_feasibility_multi_gp_wrapper(model=self.model_c, l=0)]

            return -self.compute_HVI_LCB_last_step(X)
        else:
            P = self.model.get_Y_values()
            P_cur = (-np.concatenate(P,axis=1)).tolist()

            HV0_func = hypervolume(P_cur)
            HV0 = HV0_func.compute(ref_point=self.ref_point)
            HV_new = np.zeros(X.shape[0])

            for i in range(len(X)):
                x = np.atleast_2d(X[i])
                ## Hypervolume computations
                mu = -self.model.posterior_mean(x)
                var = self.model.posterior_variance(x, noise=False)
                y_lcb = mu - self.alpha * np.sqrt(var)
                y_lcb = np.transpose(y_lcb)
                P_new = (-np.concatenate(P,axis=1)).tolist()
                P_new = np.vstack([P_new, y_lcb])
                hv_new =hypervolume(P_new.tolist())
                try:
                    HV_new[i] = hv_new.compute(ref_point = self.ref_point)
                except:
                    print("warwning! points outside reference")
                    HV_new[i] =0
            HVI = HV_new - HV0
            HVI[HVI < 0.0] = 0.0
            return HVI

    def ref_point_computation(self):
        ref_point_vector = []
        for m in range(self.model.output_dim):
            dim_m_conf_bound = self.model_mean_var_wrapper(j = m, alpha_ref_point=0.0)
            inner_opt_x, inner_opt_val = self.optimizer.optimize(f=dim_m_conf_bound)
            ref_point_vector.append(-float(inner_opt_val))
        ref_point_vector = np.array(ref_point_vector) + 0.1

        return list(ref_point_vector)

    def model_mean_var_wrapper(self, j, alpha_ref_point=None):
        if alpha_ref_point is not None:
            alpha =alpha_ref_point
        else:
            alpha = self.alpha
        def model_mean_var(X):
            X = np.atleast_2d(X)
            mu = -self.model.posterior_mean_builder(X, j)
            var = self.model.posterior_var_builder(X, j)
            y_lcb = mu - alpha * np.sqrt(var)
            # if self.constraints_flag:
            #     Fz = self.probability_feasibility_multi_gp_wrapper(self.model_c)(X).reshape(-1, 1)
            #     y_lcb = Fz.reshape(-1) * y_lcb.reshape(-1)
            return  -np.array(y_lcb).reshape(-1)
        return model_mean_var

    def model_mean_var_grad_wrapper(self, j):
        alpha = 0#self.alpha
        def model_mean_var_grad(X):
            X = np.atleast_2d(X)

            mu = -self.model.posterior_mean_builder(X, j)

            grad_mu = self.model.posterior_mean_grad_builder(X, j)
            grad_mu = -np.array(grad_mu).reshape(-1)
            grad_var = self.model.posterior_variance_grad_builder(X, j)
            grad_var = np.array(grad_var).reshape(-1)
            var = self.model.posterior_var_builder(X, j)
            try:
                recip_var = np.reciprocal(np.sqrt(var))
            except:
                print("went to except, prob problems with reciprocal of var...", var)
                var = 1.0e-21
                recip_var = np.reciprocal(np.sqrt(var))

            y_lcb_grad = grad_mu - alpha * (1.0/2.0)* recip_var * grad_var

            y_lcb = mu - self.alpha * np.sqrt(var)

            return  -np.array(y_lcb).reshape(-1), -np.array(y_lcb_grad).reshape(1,-1)
        return model_mean_var_grad

    def update_current_best(self):
        n_observations = self.model.get_X_values().shape[0]
        if n_observations > self.counter:
            print("updating current best..........")
            self.counter = n_observations
            self.current_max_xopt, self.current_max_value = self._compute_current_max()
        assert self.current_max_value.reshape(-1) is not np.inf; "error ocurred updating current best"

    def _compute_current_max(self):
        def current_func(X_inner):
            mu = -self.model.posterior_mean(X_inner)[0]
            mu = mu.reshape(-1, 1)
            pf = self.probability_feasibility_multi_gp(X_inner, self.model_c).reshape(-1, 1)
            return -(mu * pf)
        inner_opt_x, inner_opt_val = self.optimizer.optimize(f=current_func, f_df=None, num_samples=1000,verbose=False)
        # print("inner_opt_x, inner_opt_val",inner_opt_x, inner_opt_val)
        return inner_opt_x,-inner_opt_val

    def probability_feasibility_multi_gp_wrapper(self, model, l=0):
        def probability_feasibility_multi_gp(x):
            # print("model",model.output)
            x = np.atleast_2d(x)
            Fz = []
            for m in range(model.output_dim):
                Fz.append(self.probability_feasibility(x, model.output[m], l=0))
            Fz = np.product(Fz, axis=0)
            return Fz
        return probability_feasibility_multi_gp


    def probability_feasibility(self, x, model, l=0):

        mean = model.posterior_mean(x)
        var = model.posterior_variance(x, noise=False)
        std = np.sqrt(var).reshape(-1, 1)

        mean = mean.reshape(-1, 1)

        norm_dist = norm(mean, std)
        Fz = norm_dist.cdf(l)

        return Fz.reshape(-1, 1)

