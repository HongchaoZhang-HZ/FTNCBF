from SensorFaults import *
from SNCBF_Synth import *


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
        self.__SNCBF_Init__()
        # Initialize EKF list
        self.EKF_list = []
        # Initialize FT parameters
        self.gamma_list = gamma_list
        self.FTEst_list = []

    def __SNCBF_Init__(self):
        # Define SNCBFs
        for i in range(self.fault_list.num_faults):
            # Todo: adjustment The adjustment allow users to modify the arch of each NN.
            SNCBF = SNCBF_Synth(self.arch,
                                self.act_layer,
                                self.case, verbose=self.verbose)
            self.SNCBF_list.append(SNCBF)

    def __EKF_Init__(self):
        EKF_list = []
        # Define SNCBFs' EKF Gain
        # Todo: update SNCBF EKF Gain

    def __FTEst_Init__(self):
        FTEst_list = []
        # Define SNCBFs' EKF Gain
        # Todo: update SNCBF EKF Gain

    def train(self, num_epoch, num_restart):
        for SNCBF in self.SNCBF_list:
            # Train SNCBFs
            SNCBF.train(num_epoch=num_epoch, num_restart=num_restart, warm_start=False)
            # veri_result, num = SNCBF.veri.proceed_verification()


sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
fault_list = FaultPattern(sensor_list,
                          fault_target=[[1], [2, 3]],
                          fault_value=[[0.1], [0.15, 2]])
ObsAvoid = ObsAvoid()
gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01]
FTFramework(arch=[32, 32], act_layer=[True, True], case=ObsAvoid,
            sensors=sensor_list, faults=fault_list,
            gamma_list=gamma_list, verbose=True)
