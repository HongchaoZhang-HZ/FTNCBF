# from SNCBF_Synth import *
from FTEst import *
# from Controller import NCBFCtrl
from CRNN_v1 import CRNN
from Cases.ObsAvoid import ObsAvoid
from torch import optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
class Conflict_Resolution():
    '''
    Define conflict resolution to exclude wrong estimates
    The mechanism of the proposed conflict resolution is as follows:
        - compute estimates based on different FT-Estimators, i.e., EKF
        - construct SNCBF for each possible fault patterns
        - compute control input u to satisfy each SNCBF
    '''
    def __init__(self,
                 sensor_list: SensorSet,
                 fault_list: FaultPattern,
                 case):
        # self.SNCBF_list = SNCBF_list
        self.sensor_list = sensor_list
        self.fault_list = fault_list
        self.case = case
        self.obsMatrix = self.sensor_list.obs_matrix
        self.fault_masks = self.fault_list.fault_mask_list
        # self.controller = controller
        # self.backupctrl_list = backupctrl_list
        # self.model = CRNN([32, 32], [True, True], [[-2,2],[-2,2],[-2,2]])
        self.model = CRNN(self.sensor_list.num_sensors, self.fault_list.num_faults+1)


    def Generate_sensor_readings(self, size: int = 100, sensor_noise: list = [0.001, 0.002, 0.0015, 0.001, 0.01]) -> torch.Tensor:
        '''
        Generate sensor readings
        :param size: the number of samples on each dimension
        :return: a mesh grid torch data
        '''
        # state_space = self.case.DOMAIN
        state_space = []
        for sensors in self.sensor_list.sensor_list:
            state_space.append(self.case.DOMAIN[sensors.obs])
        shape = []
        for _ in range(len(state_space)):
            shape.append(size)
        cell_length = (state_space[0][1] - state_space[0][0]) / size
        raw_data = []
        for i in range(len(state_space)):
            data_element = torch.linspace(state_space[i][0] + cell_length/2, state_space[i][1] - cell_length/2, shape[0])
            noise = torch.normal(mean=torch.zeros(shape[0]),std=sensor_noise[i]),
            raw_data.append(data_element + noise[0])
        raw_data_grid = torch.meshgrid(raw_data)
        X_raw_list = []
        for raw_item in raw_data_grid:
            X_raw_list.append(raw_item.flatten())
        X_raw_stack = torch.stack(X_raw_list)
        return X_raw_stack

    def generate_attack_label(self, sensor_readings):
        num_data = sensor_readings[0].shape[0]
        y_raw_stack = torch.zeros(sensor_readings.shape[-1])

        # Simulate attack on sensor readings and generate labels
        X_data = sensor_readings.type(dtype=torch.float)
        Y_label = y_raw_stack
        for reading_idx in range(num_data):
            x_data_item = X_data[:, reading_idx]
            attack_probability = 0.5  # Adjust as needed
            # Generate a random flag for attack or not
            attack_flag = torch.rand(1) < attack_probability
            if attack_flag:
                # Uniformly randomly choose a sensor index to deploy attacks
                attack_idx = torch.randint(0, self.fault_list.num_faults, (1,))
                # Simulate attack on sensor readings
                fault = self.fault_list.fault_list[attack_idx]
                for idx in range(len(fault.target)):
                    attack_sig = fault.value[idx]
                    x_data_item[fault.target[idx]] += torch.tensor(attack_sig)
                # If train with target use this:
                # y_label = torch.tensor(fault.target)
                # If train with fault use this
                y_label = torch.tensor(attack_idx+1)
                X_data[:, reading_idx] = x_data_item
                Y_label[reading_idx] = y_label

        return X_data, Y_label

    def train(self):
        # Set up TensorBoard logger
        logger = pl.loggers.TensorBoardLogger("logs", name="sensor_classifier")
        trainer = pl.Trainer(accelerator='cpu', max_epochs=10, log_every_n_steps=5, logger=logger)

        X_cleandata = self.Generate_sensor_readings(16)
        X_data, Y_label = self.generate_attack_label(X_cleandata)

        X_train, X_val, Y_train, Y_val = train_test_split(X_data.transpose(0,1), Y_label.to(torch.long),
                                                          test_size=0.4, random_state=42)
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        trainer.fit(self.model, train_loader, val_loader)
        return


sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
fault_list = FaultPattern(sensor_list,
                          fault_target=[[1], [2, 3]],
                          fault_value=[[0.1], [0.15, 2]])
ObsAvoid = ObsAvoid()
CR = Conflict_Resolution(sensor_list, fault_list, ObsAvoid)
for i in range(5):
    CR.train()
X_cleandata = CR.Generate_sensor_readings(16)
X_data, Y_label = CR.generate_attack_label(X_cleandata)
X_train, X_val, Y_train, Y_val = train_test_split(X_data.transpose(0,1), Y_label.to(torch.long),
                                                          test_size=0.4, random_state=42)
from sklearn.metrics import accuracy_score
test_size = int(1e4)
logits = CR.model(X_val[:test_size,:])
predicted_class = torch.argmax(logits, dim=1)
accuracy = accuracy_score(Y_val.to(torch.long)[:test_size], predicted_class)
print(accuracy)

