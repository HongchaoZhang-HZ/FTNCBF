import torch
from Cases.Case import case

from FTEst import FTEst
from SNCBF_Synth import *
from Cases.ObsAvoid import ObsAvoid
from SensorFaults import *
class NCBFCtrl:
    def __init__(self, DIM, SNCBF_list, FTEst_list, case: object, gamma_list):
        self.DIM = DIM
        self.SNCBF_list = SNCBF_list
        self.FTEst_list = FTEst_list
        self.case = case
        self.fx = self.case.f_x
        self.gx = self.case.g_x
        self.gamma_list = gamma_list
        self.grad_max_list = self.grad_max_Init()

    def compute_u(self, state):
        # TODO: unconstrained minimization as a baseline
        # minimize ||u||
        u = 0
        return u

    def dbdx(self, x):
        dbdx = 0
        return dbdx

    def d2bdx2(self, x):
        d2bdx2 = 0
        return d2bdx2

    # def grad_max_Init(self):
    #     return

    def solo_SCBF_condition(self, x, EKFGain, obsMatrix, gamma):
        dbdx = self.dbdx(x)
        # stochastic version
        fx = self.fx(torch.Tensor(x).reshape([1, self.DIM])).numpy()
        dbdxf = dbdx @ fx
        EKF_term = dbdx @ EKFGain @ obsMatrix
        # stochastic_term = gamma * np.linalg.norm(EKF_term) - grad_max * gamma
        stochastic_term = gamma * np.linalg.norm(EKF_term)
        # TODO: second order derivative

        return dbdxf - stochastic_term

    def multi_SCBF_conditions(self, x):
        cons = []
        for SNCBF_idx in range(self.num_SNCBF):
            # Update observation matrix
            obsMatrix = self.fault_list.fault_mask_list[SNCBF_idx]
            # Update EKF gain
            EKFGain = self.FTEKF_gain_list[SNCBF_idx]
            # Compute SCBF constraint
            SCBF_cons = self.solo_SCBF_condition(x, EKFGain, obsMatrix,
                                                 self.gamma_list[SNCBF_idx])

    def CBF_based_u(self, state):
        # compute based on self.CBF
        dbdx = self.dbdx(x)
        affine_gain = dbdx @ self.gx(x)
        # TODO: constrained minimization
        # minimize ||u||
        # constraint: affine_gain @ u + self.SCBF_conditions(x)

        return

sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
fault_list = FaultPattern(sensor_list,
                          fault_target=[[1], [2, 3]],
                          fault_value=[[0.1], [0.15, 2]])
ObsAvoid = ObsAvoid()
gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01]
SNCBF0 = SNCBF_Synth([32, 32], [True, True], ObsAvoid, verbose=True)
SNCBF0.model.load_state_dict(torch.load('Trained_model/SNCBF/SNCBFGood/SNCBF_Obs0.pt'), strict=True)
SNCBF1 = SNCBF_Synth([32, 32], [True, True], ObsAvoid, verbose=True)
SNCBF1.model.load_state_dict(torch.load('Trained_model/SNCBF/SNCBFGood/SNCBF_Obs1.pt'), strict=True)
FTEst_list = FTEst(None, sensor_list, fault_list)
ctrl = NCBFCtrl(ObsAvoid.DIM, [SNCBF0, SNCBF1], FTEst_list, ObsAvoid, gamma_list)
