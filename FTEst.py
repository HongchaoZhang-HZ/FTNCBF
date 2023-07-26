from SensorFaults import *
from EKF import *

class FTEst():
    '''
    Define fault-tolerant estimators.
    The base FT-Estimator is: FT-EKF.
    FT-EKF maps sensor readings to state estimates.
    Input: Faults, Sensors
    '''
    def __init__(self, gamma_list, FaultSet, SensorSet):
        # Initialize sensors and faults
        self.Sensors = SensorSet
        self.num_sensors = SensorSet.num_sensors
        self.Faults = FaultSet
        self.num_faults = FaultSet.num_faults
        if gamma_list is None:
            gamma_list = 0.001 * np.ones(self.num_faults)
        self.s_blank_idx = torch.ones(self.num_sensors)

    @property
    def sensor_index(self):
        # use itertools to create a mask
        return

    @property
    def Fault_Pattern(self):
        for fault_i in self.Faults:
            fault_i = self.s_blank_idx

    def EKF(self):
        landmarks = np.array([[5, 10, 0.5], [10, 5, 0.5], [15, 15, 0.5]])

        ekf = run_localization(
            landmarks, std_vel=0.1, std_steer=np.radians(1),
            std_range=0.3, std_bearing=0.1)
        print('Final P:', ekf.P.diagonal())
        return ekf.K