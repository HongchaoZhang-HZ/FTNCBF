import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class Sensor:
    obs: int
    noise: float


class SensorSet:
    num_sensors: int

    def __init__(self, sensor_obs=None, sensor_noise=None):
        self.obs_matrix = None
        if sensor_noise is None:
            sensor_noise = [0.01, 0.01, 0.01]
        if sensor_obs is None:
            sensor_obs = [0, 1, 2]
        self.sensor_list = []
        self.state_dim = np.max(sensor_obs) + 1
        self.sensor_set_init(sensor_obs, sensor_noise)
        self.num_sensors = len(self.sensor_list)
        self.obs_matrix_init()

    def sensor_set_init(self, sensor_obs, sensor_noise):
        sensor_list = []
        for item in range(len(sensor_obs)):
            obs = sensor_obs[item]
            sigma = sensor_noise[item]
            sensor_list.append(Sensor(obs, sigma))
        self.sensor_list = sensor_list

    def obs_matrix_init(self):
        obs_matrix = np.zeros([self.num_sensors, self.state_dim])
        for idx in range(self.num_sensors):
            obs_matrix[idx][self.sensor_list[idx].obs] = 1
        self.obs_matrix = obs_matrix

    @property
    def view_sensors(self):
        return self.sensor_list


@dataclass
class Fault:
    idx: int
    target: list
    value: list


class FaultPattern:
    def __init__(self, sensors, fault_target=None, fault_value=None):
        self.fault_mask_list = []
        self.fault_list = []
        if fault_target is None:
            fault_target = [[0], [1, 2]]
        if fault_value is None:
            fault_value = [[0.01], [0.01, 0.01]]
        self.Sensors = sensors
        self.fault_list_init(fault_target, fault_value)
        self.num_faults = len(self.fault_list)
        self.fault_mask_init()

    def fault_list_init(self, fault_target, fault_value):
        fault_list = []
        for fidx in range(len(fault_target)):
            target = fault_target[fidx]
            attack = fault_value[fidx]
            fault_list.append(Fault(fidx, target, attack))
        self.fault_list = fault_list

    def fault_mask_init(self):
        for fault in self.fault_list:
            fault_mask_matrix = self.Sensors.obs_matrix.copy()
            # If there is no fault, fault_mask = self.Sensors.obs_matrix
            for target in fault.target:
                fault_mask_matrix[target][self.Sensors.sensor_list[target].obs] = 0
            self.fault_mask_list.append(fault_mask_matrix)


# sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
# fault_list = FaultPattern(sensor_list,
#                           fault_target=[[1], [2, 3]],
#                           fault_value=[[0.1], [0.15, 2]])
