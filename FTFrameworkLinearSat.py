import numpy as np
import torch

from SensorFaults import *
from SoloSNCBFLinearSat import *
from FTEst import *
from Controller import NCBFCtrl
import linearKF
from Cases.LinearSat import LinearSat
# from ConflictResolution import Conflict_Resolution
import itertools

class FTFramework:
    """
    Define initial FT-NCBF framework, the overall NCBF looks like follows:
    Sensor Faults - FTEst(EKFs) - SNCBFs - Conflict Resolution (Mini-Norm Controller)
    This framework is a foundation with EKF and hard-coded CR.
    Future work will base on this class and incorporate new FTEst and CR.
        :param arch: [list of int] architecture of the NN
        :param act_layer: [list of bool] if the corresponding layer with ReLU, then True
        :param case: Pre-defined case class, with f(x), g(x) and h(x)
        :param sensors: class SensorSet
        :param faults: class FaultPattern
        :param verbose: Flag for display or not
    """

    def __init__(self, arch, act_layer, case,
                 sensors: SensorSet,
                 fault_target: list,
                 fault_value: list,
                 sigma: list, nu: list,
                 gamma_list: list,
                 verbose=False):
        """
        FTFramework initialization
        :param arch: [list of int] architecture of the NN
        :param act_layer: [list of bool] if the corresponding layer with ReLU, then True
        :param case: Pre-defined case class, with f(x), g(x) and h(x)
        :param sensors: class SensorSet
        :param faults: class FaultPattern
        :param verbose: Flag for display or not
        """
        self.arch = arch
        self.act_layer = act_layer
        self.case = case
        # Define sensors
        self.sensor_list = sensors
        # Define faults
        self.fault_target = fault_target
        self.fault_value = fault_value
        # Define fault list with object faults
        self.fault_target_list = []
        self.fault_value_list = []
        self.fault_list = []
        # Initialization: fill list with (m 2) fault combinations
        self.__Fault_list_Init__()
        self.sigma = torch.tensor(sigma, dtype=torch.float).unsqueeze(1)
        self.nu = torch.tensor(nu, dtype=torch.float).unsqueeze(1)
        self.verbose = verbose

        # Initialize SNCBF list
        self.SNCBF_list = []
        self.num_SNCBF = None
        self.__SNCBF_Init__()
        # Initialize EKF list
        # self.EKF_list = []
        # Initialize FT parameters
        self.gamma_list = gamma_list
        self.FTEst = FTEst(None, self.sensor_list, self.fault_list)
        self.FTEKF_gain_list = self.FTEst.EKFgain_list
        self.Controller = NCBFCtrl(self.case.DIM, self.SNCBF_list,
                                   self.FTEst, self.case,
                                   sigma=self.sigma,
                                   nu=self.nu,
                                   gamma_list=self.gamma_list)
        self.BackUpCtrl_list = []
        self.__BackUp_Ctrl_Init__()
        # self.CR = Conflict_Resolution(SNCBF_list=self.SNCBF_list,
        #                               sensor_list=self.sensor_list,
        #                               fault_list=self.fault_list,
        #                               case=self.case,
        #                               controller=self.Controller,
        #                               backupctrl_list=self.BackUpCtrl_list)

    def __SNCBF_Init__(self):
        # Define SNCBFs
        for i in range(self.fault_list.num_faults):
            # Todo: adjustment The adjustment allow users to modify the arch of each NN.
            SNCBF = SNCBF_Synth(self.arch, self.act_layer,
                                self.case,
                                sigma=self.sigma,
                                nu=self.nu,
                                verbose=self.verbose)
            self.SNCBF_list.append(SNCBF)
            self.num_SNCBF = len(self.SNCBF_list)

    def __Fault_list_Init__(self):
        # fault_target=[{1}, {2}]
        # fault_value=[{0.1}, {0.15}]
        target_comb = list(itertools.combinations(self.fault_target, 2))
        # value_comb = list(itertools.combinations(self.fault_value, 2))
        total_tar = []
        # total_val = []
        for com_idx in range(len(target_comb)):
            target_set = target_comb[com_idx][0]
            # value_set = value_comb[com_idx][0]
            for item_idx in range(len(target_comb[com_idx])):
                target_set = target_set.union(target_comb[com_idx][item_idx])
                # value_set = value_set.union(value_comb[com_idx][item_idx])
            total_tar.append(target_set)
            # total_val.append(value_set)

        # initiate fault target list
        fault_target_list = []
        fault_value_list = []
        for item in self.fault_target:
            # get fault list items of each fault
            flist = list(item)
            # make it a list
            fault_target_list.append(flist)
            # append corresponding values
            fault_value_list.append([self.fault_value[self.fault_target.index({i})] for i in item])
        for item in total_tar:
            flist = list(item)
            fault_target_list.append(flist)
            fault_value_list.append([self.fault_value[self.fault_target.index({i})] for i in item])
        self.fault_target_list = fault_target_list
        self.fault_list = FaultPattern(self.sensor_list,
                                       fault_target=fault_target_list,
                                       fault_value=fault_value_list)

    def __BackUp_Ctrl_Init__(self):
        # import itertools
        # lst = list(itertools.product([0, 1], repeat=self.num_SNCBF))
        # backup_list = []
        # for item in lst:
        #     item = list(item)
        #     backup_list.append(item)

        BackUpCtrl_list = []
        # TODO: change BackUp_List to self.BackUp_List when init is finished
        BackUp_List = [[1,0], [0,1]]
        for idx in range(len(BackUp_List)):
            switch = BackUp_List[idx]
            backup_scbf_list = []
            for item in range(len(switch)):
                if item:
                    backup_scbf_list.append(self.SNCBF_list[idx])
            BackupCtrl = NCBFCtrl(self.case.DIM, backup_scbf_list,
                                  self.FTEst, self.case,
                                  self.sigma, self.nu,
                                  self.gamma_list)
            BackUpCtrl_list.append(BackupCtrl)
        self.BackUpCtrl_list = BackUpCtrl_list

    def train(self, num_epoch, num_restart, alpha1, alpha2):
        for SNCBF_idx in range(self.num_SNCBF):
            # Update observation matrix
            self.SNCBF_list[SNCBF_idx].update_obs_matrix_c(torch.Tensor(self.fault_list.fault_mask_list[SNCBF_idx]))
            # Update EKF gain
            self.SNCBF_list[SNCBF_idx].update_EKF_gain(torch.Tensor(self.FTEKF_gain_list[SNCBF_idx]))
            # Train SNCBFs
            self.SNCBF_list[SNCBF_idx].train(num_epoch=num_epoch, num_restart=num_restart,
                                             alpha1=alpha1, alpha2=alpha2, warm_start=False)
            # veri_result, num = SNCBF.veri.proceed_verification()

    def FTNCBF_Framework(self, T, dt, x0):
        # Define initial state x
        # x0 = np.array([[1.0,1.0,0.0]])
        x = x0
        traj = []
        u_list = []
        traj.append(x)
        co_flag = []
        est_list = []
        # TODO: put these in self.CR.Resolution(res.success)
        for t in range(T):
            if t>=50:
                x[0, 0] += -0.5
            # NCBF control optimizer output
            res = self.Controller.CBF_based_u(x)
            co_flag.append(res.success)
            # STEP 1 compute u that satisfy all SNCBF cons
            if res.success:
                u = res.x
            # if the result has a false feasibility flag,
            # then go step 2
            else:
                # TODO: tempt_flist = self.BackUp_list
                tempt_flist = self.fault_list.fault_list
                # STEP 2 Get difference
                # 2-1. Get estimates
                est = []
                diff_list = []
                double_check_diff_list = []
                for idx in range(2):
                    # TODO: replace number 2 with self.fault_list.num_faults
                    est.append(self.FTEst.FTEst_list[idx].x)
                    diff_list.append(np.linalg.norm(self.FTEst.FTEst_list[idx].x - x.transpose()))
                    # TODO: replace 3-2 with (backuplist length - self.fault_list.num_faults)
                    dc_diff_list = np.linalg.norm(np.array([self.FTEst.FTEst_list[3-2].x]) - self.FTEst.FTEst_list[idx].x)
                    double_check_diff_list.append(np.min(dc_diff_list))

                est_list = np.array(est)
                diff_array = np.array(diff_list)
                double_check_diff_array = np.array(double_check_diff_list)
                # 2-2. Get Difference
                if np.min(double_check_diff_array) >= self.gamma_list[np.argmin(double_check_diff_array)]:
                    # update fault_list
                    tempt_flist.remove(self.fault_list.fault_list[np.min(double_check_diff_array)])
                    NCBFCtrl(ObsAvoid.DIM, tempt_flist, FTEst, ObsAvoid, gamma_list)
                # TODO: fix this
                safe_idx = self.fault_list.fault_list.index(tempt_flist)
                step2res = self.BackUpCtrl_list[safe_idx].compute_u(x)
                u = step2res.x

                step2res = False
                # if the result has a false feasibility flag,
                # then go step 3
                if not step2res:
                    max_residual_idx = np.argmax(diff_array)
                    tempt_flist.remove(self.fault_list.fault_list[max_residual_idx])
                    # TODO: fix this
                    safe_idx = self.fault_list.fault_list.index(tempt_flist)
                    step3res = self.BackUpCtrl_list[safe_idx].compute_u(x)
                    u = step3res.x
            # Post controller steps

            # update state x in a discrete time manner
            x = torch.tensor(x)
            x = x + dt * (self.case.f_x(x) + (self.case.g_x(x) @ u).unsqueeze(1)).transpose(0, 1)
            x = x.numpy()
            for item in self.FTEst.FTEst_list:
                item.x = x
            traj.append(x)
            u_list.append(u)
            # est_list.append(self.FTEst.FTEst_list[0].x)
        return traj, u_list, co_flag

    def Baseline(self, T, dt, x0):
        # Define initial state x
        # x0 = np.array([[1.0,1.0,0.0]])
        x = x0
        traj = []
        u_list = []
        traj.append(x)
        co_flag = []
        est_list = []
        # TODO: put these in self.CR.Resolution(res.success)
        for t in range(T):
            if t>=50:
                x[0, 0] += -0.5
            # NCBF control optimizer output
            res = self.Controller.CBF_based_u(x)
            co_flag.append(res.success)
            # STEP 1 compute u that satisfy all SNCBF cons
            u = res.x

            x = torch.tensor(x)
            x = x + dt * (self.case.f_x(x) + (self.case.g_x(x) @ u).unsqueeze(1)).transpose(0, 1)
            x = x.numpy()
            for item in self.FTEst.FTEst_list:
                item.x = x
            traj.append(x)
            u_list.append(u)
            # est_list.append(self.FTEst.FTEst_list[0].x)
        return traj, u_list, co_flag



sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
fault_list = FaultPattern(sensor_list,
                          fault_target=[[1], [2]],
                          fault_value=[[0.1], [0.15]])
LinearSat = LinearSat()
sensor_list = SensorSet([0, 1, 1, 2, 2, 3, 4, 5],
                        [0.001, 0.002, 0.0015, 0.001, 0.01, 0.001, 0.01, 0.001])
fault_list = FaultPattern(sensor_list,
                          fault_target=[[1], [3]],
                          fault_value=[[0.1], [0.15]])
gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01, 0.001, 0.01, 0.001]
newCBF = SNCBF_Synth([128, 128], [True, True], LinearSat,
                     sigma=[0.001, 0.001, 0.001, 0.00001, 0.0001, 0.00001, 0.00001, 0.00001],
                     nu=[0.001, 0.001, 0.001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
                     gamma_list=gamma_list,
                     verbose=True)
newCBF.model.load_state_dict(torch.load('Trained_model/NCBF/lSat_fixnt.pt'))
K1, K3 = linearKF.linearKF()

MU = 3.986e14
a = 500e3
n = sqrt(MU/a**3)
# A = np.array([[1, 0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0, 0],
#               [3 * n ** 2, 0, 0, 0, 2 * n, 0],
#               [0, 0, 0, -2 * n, 0, 0],
#               [0, 0, -n ** 2, 0, 0, 0]])
dt = 1e-2
# A = np.array([[1, 0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0, 0],
#               [0.03015, 0, 0, 0.9998, 0.002, 0],
#               [-0.000301, 0, 0, -0.002, 0.9998, 0],
#               [0, 0, -0.01005, 0, 0, 1]])
A = np.array([[1, 0, 0, dt, 0, 0],
              [0, 1, 0, 0, dt, 0],
              [0, 0, 1, 0, 0, dt],
              [0.03015, 0, 0, 0.9998, 0.002, 0],
              [-0.000301, 0, 0, -0.002, 0.9998, 0],
              [0, 0, -0.01005, 0, 0, 1]])
# dt = 1e-3
# B = np.array([[0, 0, 0],
#               [0, 0, 0],
#               [0, 0, 0],
#               [1, 0, 0],
#               [0, 1, 0],
#               [0, 0, 1]])
B = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0.009999, 0.0001, 0],
              [-0.0001, 0.009999, 0],
              [0, 0, 0.01]])

C1 = np.array([[1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

C3 = np.array([[1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

# Define initial state estimate and covariance
x_hat = np.zeros((6, 1))  # Initial state estimate
P = np.eye(6)  # Initial state covariance

# Define process and measurement noise covariances
Q = np.eye(6)  # Process noise covariance (adjust as needed)
R = np.eye(8)  # Measurement noise covariance (adjust as needed)

# Simulated measurements (replace with your measurements)
num_time_steps = 50
u = np.zeros((3, 1))  # Control input (if any)
# tensor([ 1.7551, -1.7449,  1.7551, -1.7449, -1.7449,  1.7551])
init_x = torch.Tensor([-0.7410,  1.2590, -0.2410,  1.7590, -1.2410,  1.7590]).numpy()
x_hat_minus = np.expand_dims(init_x, -1)
# x_hat_minus = np.array([[1.6760], [-1.6573], [1.6760], [-1.6573], [-1.6573], [1.6760]])
x_hat1 = x_hat_minus
x_hat3 = x_hat_minus
P1 = np.eye(6)
P3 = np.eye(6)
traj = []
# Kalman filter loop
for k in range(num_time_steps):
    # Prediction step

    traj.append(x_hat_minus)
    P1_minus = np.dot(np.dot(A, P1), A.T) + Q
    P3_minus = np.dot(np.dot(A, P3), A.T) + Q

    measurement = C1 @ x_hat_minus + np.expand_dims(np.random.normal(0, 0.001, 8), -1)
    attack = np.array([[0],[0],[0],np.random.normal(-1, 0.1, 1),[0],[0],[0],[0]])
    measurement = measurement + attack
    # Update step
    K1 = np.dot(np.dot(P1_minus, C1.T), np.linalg.inv(np.dot(np.dot(C1, P1_minus), C1.T) + R))
    x_hat1 = x_hat_minus + K1 @ (measurement - C1 @ x_hat_minus)
    P1 = np.dot((np.eye(6) - np.dot(K1, C1)), P1_minus)
    K3 = np.dot(np.dot(P3_minus, C3.T), np.linalg.inv(np.dot(np.dot(C3, P3_minus), C3.T) + R))
    x_hat3 = x_hat_minus + K3 @ (measurement - C3 @ x_hat_minus)
    P3 = np.dot((np.eye(6) - np.dot(K3, C3)), P3_minus)
    ekf_gain = [K1, K3]
    C = newCBF.c

    trace_term_list = []
    stochastic_term_list = []
    cons = tuple()
    if np.linalg.norm(x_hat3-x_hat1)<0.1:
        xlist = [x_hat1, x_hat3]
    else:
        xlist = [x_hat3]
    for i in range(len(xlist)):
        x = xlist[i]
        grad_vector = newCBF.get_grad(x.transpose())[0].numpy()
        # dbdxfx = grad_vector @ newCBF.case.f_x(torch.Tensor(x.transpose())).numpy()
        dbdxfx = (grad_vector @ A @ x)[0]
        # dbdxgx = grad_vector @ newCBF.case.g_x(torch.Tensor(x.transpose())).numpy()
        dbdxgx = (grad_vector @ B)[0]
        EKF_term = grad_vector @ ekf_gain[i] @ C[i].numpy()
        stochastic_term = -newCBF.gamma[i] * np.linalg.norm(EKF_term)
        stochastic_term_list.append(stochastic_term)

        hess = newCBF.get_hessian(torch.Tensor(x.transpose()))
        second_order_term = newCBF.nu.transpose(0, 1).numpy() @ ekf_gain[i].transpose() \
                            @ hess.numpy() @ ekf_gain[i] @ newCBF.nu.numpy()
        trace_term = second_order_term.trace()
        trace_term_list.append(trace_term)
        SoloCBFCon = lambda u: (dbdxgx @ u + dbdxfx)*dt + newCBF.forward(torch.Tensor(x.transpose())).detach().numpy()[0][0] + stochastic_term + trace_term - 0.002
        SoloOptCBFCon = NonlinearConstraint(SoloCBFCon, 0, np.inf)
        cons = cons + (SoloOptCBFCon,)

    u0 = np.array([0.001, -0.001, -0.001])
    def fcn(u, dbdxgx):
        np.sum(u**2)
        return 0
    res = minimize(fcn, u0, args=dbdxgx, constraints=SoloOptCBFCon)
    u = np.expand_dims(res.x, -1)
    # u = np.expand_dims(u0, -1)
    x_hat_minus = A @ x_hat_minus + B @ u  # u is the control input (if any)

trajr = []
for i in traj:
    r = sqrt(np.sum(i[:3] ** 2))
    trajr.append(r)
for i in traj:
    print(newCBF.forward(torch.Tensor(i).transpose(0, 1)))
# binput = newCBF.generate_data(8)
# bout = newCBF.forward(binput)
# bout.argmax()
# for i in range(binput.shape[0]):
#     if sqrt(np.sum(binput[i][:3].numpy() ** 2)) < 1.5 and sqrt(np.sum(binput[i][:3].numpy() ** 2)) > 0.25:
#         if newCBF.forward(binput[i]) > 0:
#             bout.append(newCBF.forward(binput[i]))
#             if newCBF.forward(binput[i]) > 0.007:
#                 print(i)
