from SNCBF_Synth import *
from FTEst import *
from Controller import NCBFCtrl
class Conflict_Resolution:
    '''
    Define conflict resolution to exclude wrong estimates
    The mechanism of the proposed conflict resolution is as follows:
        - compute estimates based on different FT-Estimators, i.e., EKF
        - construct SNCBF for each possible fault patterns
        - compute control input u to satisfy each SNCBF
    '''
    def __init__(self, SNCBF_list,
                 sensor_list: SensorSet,
                 fault_list: FaultPattern,
                 controller_list: list):
        self.SNCBF_list = SNCBF_list
        self.sensor_list = sensor_list
        self.fault_list = fault_list
        self.obsMatrix = self.sensor_list.obs_matrix
        self.fault_masks = self.fault_list.fault_mask_list
        self.controller_list = controller_list

    def Resolution(self):
        return



# sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
# fault_list = FaultPattern(sensor_list,
#                           fault_target=[[1], [2, 3]],
#                           fault_value=[[0.1], [0.15, 2]])
# ObsAvoid = ObsAvoid()
# gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01]
# CR = Conflict_Resolution()