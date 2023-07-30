import numpy as np

from SensorFaults import *
from SNCBF_Synth import *
from FTEst import *
from Controller import NCBFCtrl
from ConflictResolution import Conflict_Resolution

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
                 faults: FaultPattern,
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
        self.sensor_list = sensors
        self.fault_list = faults
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
                                   self.gamma_list)
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
            SNCBF = SNCBF_Synth(self.arch,
                                self.act_layer,
                                self.case, verbose=self.verbose)
            self.SNCBF_list.append(SNCBF)
            self.num_SNCBF = len(self.SNCBF_list)

    def __Fault_list_Init__(self):
        import itertools
        lst = list(itertools.product([0, 1], repeat=self.fault_list.num_faults))
        backup_list = []
        BackUp_Faultlist = []
        for item in lst:
            item = list(item)
            backup_list.append(item)
        # Scan all fault combinations
        for idx in range(len(backup_list)):
            switch = backup_list[idx]
            BackUp_Fault = []
            # Combine faults
            for item in range(len(switch)):
                if item:
                    BackUp_Fault.append(self.SNCBF_list[idx])
        # TODO: make backup list automatically update itself based on faults
        FaultPattern(sensor_list,
                     fault_target=[[1], [2], [1, 2]],
                     fault_value=[[0.1], [0.15], [0.1, 0.15]])

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
        x0 = np.array([[1,1,0]])
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
            x = x + dt * (self.case.f_x(x) + self.case.g_x(x) @ u)

        return


sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
fault_list = FaultPattern(sensor_list,
                          fault_target=[[1], [2], [1,2]],
                          fault_value=[[0.1], [0.15], [0.1, 0.15]])
ObsAvoid = ObsAvoid()
gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01]
FTNCBF = FTFramework(arch=[32, 32], act_layer=[True, True], case=ObsAvoid,
                     sensors=sensor_list, faults=fault_list,
                     gamma_list=gamma_list, verbose=True)
# FTNCBF.train(num_epoch=10, num_restart=2)
FTNCBF.FTNCBF_Framework(100, dt, np.array([[1,1,0]]))