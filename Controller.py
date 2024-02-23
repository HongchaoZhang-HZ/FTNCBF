import numpy as np
import torch
from Cases.Case import case
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from torch.autograd.functional import hessian

from FTEst.FTEst import FTEst
from SNCBF_Synth import *
from Cases.ObsAvoid import ObsAvoid
from FTEst.SensorFaults import *

class NCBFCtrl:
    def __init__(self, DIM, SNCBF_list, FTEst,
                 case: object,
                 sigma, nu,
                 gamma_list):
        self.DIM = DIM
        self.SNCBF_list = SNCBF_list
        self.num_SNCBF = len(SNCBF_list)
        self.FTEst = FTEst
        self.FTEKF_gain_list = self.FTEst.EKFgain_list
        self.case = case
        self.fx = self.case.f_x
        self.gx = self.case.g_x
        self.gamma_list = gamma_list
        self.sigma = sigma
        self.nu = nu
        # self.grad_max_list = self.grad_max_Init()

    def compute_u(self, x):
        def fcn(u):
            return (u**2).sum()
        # minimize ||u||
        u0 = np.array([0])
        res = minimize(fcn, u0)
        return res

    def d1bdx1(self, SNCBF, x:float):
        grad_input = torch.tensor(x, requires_grad=True)
        dbdx = torch.autograd.grad(SNCBF.model.forward(grad_input), grad_input)
        return dbdx

    def d2bdx2(self, SNCBF, x):
        # grad_input = torch.tensor([[0.0,0.0,0.0]], requires_grad=True)
        # out = FTNCBF.SNCBF_list[0].model.forward(grad_input)
        # out.backward(create_graph=True) # first order grad
        # out.backward(retain_graph=True) # second order grad


        grad_input = torch.tensor(x, dtype=torch.float, requires_grad=True)
        hessian_matrix = hessian(SNCBF.model.forward, grad_input).squeeze()

        # grad_input = torch.tensor(x, requires_grad=True)
        # d2bdx2 = torch.autograd.grad(self.d1bdx1(SNCBF, grad_input),
        #                              SNCBF.model.forward(grad_input))
        return hessian_matrix

    # def grad_max_Init(self):
    #     return

    def solo_SCBF_condition(self, SNCBF, x, EKFGain, obsMatrix, gamma):
        dbdx = SNCBF.get_grad(x)
        # stochastic version
        fx = self.fx(torch.Tensor(x).reshape([1, self.DIM])).numpy()
        dbdxf = dbdx @ fx
        EKF_term = dbdx @ EKFGain @ obsMatrix
        # stochastic_term = gamma * np.linalg.norm(EKF_term) - grad_max * gamma
        stochastic_term = gamma * np.linalg.norm(EKF_term)

        # second order derivative term
        hessian = self.d2bdx2(SNCBF, x)
        second_order_term = self.nu.transpose(0, 1).numpy() @ EKFGain.transpose() \
                            @ hessian.numpy() @ EKFGain @ self.nu.numpy()

        # if second_order_term.shape == torch.Size([1]):
        #     trace_term = second_order_term.item()
        # else:
        #     trace_term = second_order_term.trace()
        trace_term = second_order_term.trace()
        return dbdxf - stochastic_term + trace_term

    def multi_SCBF_conditions(self, x):
        cons = []
        gain_list = []
        for SNCBF_idx in range(self.num_SNCBF):
            # Update observation matrix
            obsMatrix = self.FTEst.fault_list.fault_mask_list[SNCBF_idx]
            # Update EKF gain
            EKFGain = self.FTEKF_gain_list[SNCBF_idx]
            # Compute SCBF constraint
            SCBF_cons = self.solo_SCBF_condition(self.SNCBF_list[SNCBF_idx],
                                                 x, EKFGain, obsMatrix,
                                                 self.gamma_list[SNCBF_idx])
            cons.append(SCBF_cons)

            # Compute Affine Gain
            affine_gain = torch.stack(self.SNCBF_list[SNCBF_idx].get_grad(x)) @ self.gx(x)
            gain_list.append(affine_gain)
        return cons, gain_list

    def CBF_based_u(self, x):
        # compute based on self.CBF
        SCBF_cons, affine_gain = self.multi_SCBF_conditions(x)
        cons = tuple()
        for idx in range(self.num_SNCBF):
            SoloCBFCon = lambda u: (affine_gain[idx] @ u).squeeze() + (SCBF_cons[idx]).squeeze()
            SoloOptCBFCon = NonlinearConstraint(SoloCBFCon, 0, np.inf)
            cons = cons + (SoloOptCBFCon,)
        def fcn(u):
            return (u**2).sum()
        # minimize ||u||
        u0 = np.zeros(self.case.CTRLDIM)
        # minimize ||u||
        # constraint: affine_gain @ u + self.SCBF_conditions(x)
        res = minimize(fcn, u0, constraints=SoloOptCBFCon)
        return res

# sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
# fault_list = FaultPattern(sensor_list,
#                           fault_target=[[1], [2, 3]],
#                           fault_value=[[0.1], [0.15, 2]])
# ObsAvoid = ObsAvoid()
# gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01]
# SNCBF0 = SNCBF_Synth([32, 32], [True, True], ObsAvoid, verbose=True)
# SNCBF0.model.load_state_dict(torch.load('Trained_model/SNCBF/SNCBFGood/SNCBF_Obs0.pt'), strict=True)
# SNCBF1 = SNCBF_Synth([32, 32], [True, True], ObsAvoid, verbose=True)
# SNCBF1.model.load_state_dict(torch.load('Trained_model/SNCBF/SNCBFGood/SNCBF_Obs1.pt'), strict=True)
# FTEst = FTEst(None, sensor_list, fault_list)
# ctrl = NCBFCtrl(ObsAvoid.DIM, [SNCBF0, SNCBF1], FTEst, ObsAvoid, gamma_list)
# res = ctrl.CBF_based_u(np.array([[0,0,0]],dtype=np.float32))
