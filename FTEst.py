from SensorFaults import *
from RdObsEKF import *

class FTEst:
    '''
    Define fault-tolerant estimators.
    The base FT-Estimator is: FT-EKF.
    FT-EKF maps sensor readings to state estimates.
    Input: Faults, Sensors
    '''
    def __init__(self, gamma_list,
                 sensor_list: SensorSet,
                 fault_list: FaultPattern):
        # Initialize sensors and faults
        self.sensor_list = sensor_list
        self.num_sensors = self.sensor_list.num_sensors
        self.fault_list = fault_list
        self.num_faults = self.fault_list.num_faults
        if gamma_list is None:
            self.gamma_list = 0.001 * np.ones(self.num_faults)
        else:
            self.gamma_list = gamma_list
        self.FTEst_list = []
        self.EKFgain_list = []
        self.RdEKF_Init()
        self.RdEKF_Trail()

    @property
    def get_sensors(self):
        return self.sensor_list

    @property
    def get_fault_pattern(self):
        return self.fault_list

    def RdEKF_Init(self):
        for idx in range(self.num_faults):
            # TODO: make it customizable
            ekf = RdObsEKF(self.sensor_list, dt, wheelbase=0.5, std_vel=0.1,
                       std_steer=np.radians(1), std_range=0.3, std_bearing=0.1, verbose=False)
            ekf.obs_change(self.fault_list.fault_mask_list[idx])
            self.FTEst_list.append(ekf)

    def RdEKF_Trail(self):
        # TODO: make it customizable
        landmarks = np.array([[5, 10, 0.5], [10, 5, 0.5], [15, 15, 0.5]])
        for est_item in self.FTEst_list:
            est_item.run_localization(landmarks)
            self.EKFgain_list.append(est_item.K)

#
# sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
# fault_list = FaultPattern(sensor_list,
#                           fault_target=[[1], [2, 3]],
#                           fault_value=[[0.1], [0.15, 2]])
# FTE = FTEst(None, sensor_list, fault_list)