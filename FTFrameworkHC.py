import numpy as np
import torch

from SensorFaults import *
from SNCBF_Synth import *
from FTEst import *
from Controller import NCBFCtrl
from ConflictResolution import Conflict_Resolution
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
        self.CR = Conflict_Resolution(SNCBF_list=self.SNCBF_list,
                                      sensor_list=self.sensor_list,
                                      fault_list=self.fault_list,
                                      case=self.case,
                                      controller=self.Controller,
                                      backupctrl_list=self.BackUpCtrl_list)

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
        self.fault_list = FaultPattern(sensor_list,
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
        BackUp_List = [[1,0], [0,1], [1,1]]
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

    def train(self, num_epoch, num_restart):
        for SNCBF_idx in range(self.num_SNCBF):
            # Update observation matrix
            self.SNCBF_list[SNCBF_idx].update_obs_matrix_c(torch.Tensor(self.fault_list.fault_mask_list[SNCBF_idx]))
            # Update EKF gain
            self.SNCBF_list[SNCBF_idx].update_EKF_gain(torch.Tensor(self.FTEKF_gain_list[SNCBF_idx]))
            # Train SNCBFs
            self.SNCBF_list[SNCBF_idx].train(num_epoch=num_epoch, num_restart=num_restart, warm_start=False)
            # veri_result, num = SNCBF.veri.proceed_verification()

    def FTNCBF_Framework(self, T, dt, x0):
        # Define initial state x
        x0 = np.array([[1.0,1.0,0.0]])
        x = x0

        # TODO: put these in self.CR.Resolution(res.success)
        for t in range(T):
            # NCBF control optimizer output
            res = self.Controller.CBF_based_u(x)

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

        return


sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
fault_list = FaultPattern(sensor_list,
                          fault_target=[[1], [2]],
                          fault_value=[[0.1], [0.15]])
ObsAvoid = ObsAvoid()
gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01]
FTNCBF = FTFramework(arch=[32, 32], act_layer=[True, True], case=ObsAvoid,
                     sensors=sensor_list,
                     fault_target=[{1}, {2}],
                     fault_value=[[0.1], [0.15]],
                     sigma=[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
                     nu=[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
                     gamma_list=gamma_list, verbose=True)
# FTNCBF.train(num_epoch=10, num_restart=2)
FTNCBF.FTNCBF_Framework(100, dt, np.array([[1.0, 1.0, 0.0]]))