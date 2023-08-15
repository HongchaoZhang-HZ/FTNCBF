from SNCBF_Synth import *
from FTEst import *
from Controller import NCBFCtrl
from CRNN import CRNN
class Conflict_Resolution():
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
                 case, controller: NCBFCtrl,
                 backupctrl_list):
        self.SNCBF_list = SNCBF_list
        self.sensor_list = sensor_list
        self.fault_list = fault_list
        self.case = case
        self.obsMatrix = self.sensor_list.obs_matrix
        self.fault_masks = self.fault_list.fault_mask_list
        self.controller = controller
        self.backupctrl_list = backupctrl_list
        self.model = CRNN([32, 32], [True, True], [[-2,2],[-2,2],[-2,2]])

    def Resolution(self, feasible_flag):

        return

    def Generate_sensor_readings(self, size: int = 100) -> torch.Tensor:
        '''
        Generate sensor readings
        :param size: the number of samples on each dimension
        :return: a mesh grid torch data
        '''
        state_space = self.DOMAIN
        shape = []
        for _ in range(self.DIM):
            shape.append(size)
        noise = 1e-2 * torch.rand(shape)
        cell_length = (state_space[0][1] - state_space[0][0]) / size
        raw_data = []
        for i in range(self.DIM):
            data_element = torch.linspace(state_space[i][0] + cell_length/2, state_space[i][1] - cell_length/2, shape[0])
            raw_data.append(data_element)
        raw_data_grid = torch.meshgrid(raw_data)
        noisy_data = []
        for i in range(self.DIM):
            noisy_data_item = raw_data_grid[i] + noise
            # noisy_data_item = np.expand_dims(noisy_data_item, axis=self.DIM)
            noisy_data_item = noisy_data_item.reshape([torch.prod(torch.Tensor(shape),dtype=int), 1])
            noisy_data.append(noisy_data_item)
        data = torch.hstack([torch.Tensor(item) for item in noisy_data])

        return data

    def train(self, num_epoch, num_restart=10, warm_start=False):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        # define hyper-parameters
        alpha1, alpha2 = 1, 0
        # 1, 1e-8
        # Set alpha2=0 for feasibility test with Floss quickly converge to 0
        # If set alpha2 converges but does not pass the verification, then increase the sampling number.
        # This problem is caused by lack of counter examples and can be solved by introducing CE from Verifier
        rlambda = 1

        # Generate data
        size = 128
        rdm_input = self.Generate_Classivication_Data(size)
        # rdm_input = self.generate_input(shape)
        # ref_output = torch.unsqueeze(self.h_x(rdm_input.transpose(0, self.DIM)), self.DIM)
        ref_output = self.case.h_x(rdm_input).unsqueeze(1)
        normalized_ref_output = torch.tanh(10*ref_output)
        batch_length = 8**self.DIM
        training_loader = DataLoader(list(zip(rdm_input, normalized_ref_output)), batch_size=batch_length, shuffle=True)

        for self.run in range(num_restart):
            pbar = tqdm(total=num_epoch)
            veri_result = False
            for epoch in range(num_epoch):
                # Initialize loss
                running_loss = 0.0
                feasibility_running_loss = torch.Tensor([0.0])
                correctness_running_loss = torch.Tensor([0.0])
                trivial_running_loss = torch.Tensor([0.0])

                # Batch Training
                for X_batch, y_batch in training_loader:
                    model_output = self.forward(X_batch)

                    warm_start_loss = self.warm_start(y_batch, model_output)
                    correctness_loss = self.safe_correctness(y_batch, model_output, l_co=1, alpha1=alpha1, alpha2=alpha2)
                    # trivial_loss = self.trivial_panelty(ref_output, self.model.forward(rdm_input), 1)
                    trivial_loss = self.trivial_panelty(y_batch, model_output, 1)

                    grad = self.numerical_gradient(X_batch, model_output, batch_length, epsilon=0.001)
                    grad_vector = torch.vstack(grad)
                    feasibility_output = self.feasibility_loss(grad_vector, X_batch)
                    check_item = torch.max((-torch.abs(model_output)+0.2).reshape([1, batch_length]), torch.zeros([1, batch_length]))
                    # feasibility_loss = torch.sum(torch.tanh(check_item*feasibility_output))

                    # Our loss function
                    # violations = -check_item * self.feasible_violations(model_output, feasibility_output, batch_length, rlambda)
                    # Chuchu Fan loss function
                    violations = check_item * self.feasible_violations(model_output, feasibility_output, batch_length, rlambda)
                    # violations = -1 * feasibility_output - torch.max(rlambda * torch.abs(model_output.transpose(0, 1)),
                    #                                                  torch.zeros([1, batch_length]))
                    feasibility_loss = 2 * torch.sum(torch.max(violations - 1e-4, torch.zeros([1, batch_length])))
                    mseloss = torch.nn.MSELoss()
                    # loss = self.def_loss(1 * correctness_loss + 1 * feasibility_loss + 1 * trivial_loss)
                    floss = mseloss(torch.max(violations - 1e-4, torch.zeros([1, batch_length])), torch.zeros(batch_length))
                    tloss = mseloss(trivial_loss, torch.Tensor([0.0]))
                    if warm_start:
                        loss = self.warm_start(y_batch, model_output)
                    else:
                        loss = correctness_loss + feasibility_loss + tloss


                    loss.backward()
                    # with torch.no_grad():
                    #     loss = torch.max(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Print Detailed Loss
                    running_loss += loss.item()
                    feasibility_running_loss += feasibility_loss.item()
                    correctness_running_loss += correctness_loss.item()
                    trivial_running_loss += trivial_loss.item()

                    # if feasibility_running_loss <= 0.001 and correctness_loss <= 0.01:
                    #     alpha2 = 0.01
                    # else:
                    #     alpha2 = 0
                # Log details of losses
                if not warm_start:
                    self.writer.add_scalar('Loss/Loss', running_loss, self.run*num_epoch+epoch)
                    self.writer.add_scalar('Loss/FLoss', feasibility_running_loss.item(), self.run*num_epoch+epoch)
                    self.writer.add_scalar('Loss/CLoss', correctness_running_loss.item(), self.run*num_epoch+epoch)
                    self.writer.add_scalar('Loss/TLoss', trivial_running_loss.item(), self.run*num_epoch+epoch)
                # Log volume of safe region
                volume = self.compute_volume(rdm_input)
                self.writer.add_scalar('Volume', volume, self.run*num_epoch+epoch)
                # self.writer.add_scalar('Verifiable', veri_result, self.run * num_epoch + epoch)
                # Process Bar Print Losses
                pbar.set_postfix({'Loss': running_loss,
                                  'Floss': feasibility_running_loss.item(),
                                  'Closs': correctness_running_loss.item(),
                                  'Tloss': trivial_running_loss.item(),
                                  'PVeri': str(veri_result),
                                  'Vol': volume.item()})
                pbar.update(1)
                scheduler.step()


            pbar.close()
            if feasibility_running_loss <= 0.0001 and not warm_start:
                try:
                    veri_result, num = self.veri.proceed_verification()
                except:
                    pass
            # if veri_result:
            #     torch.save(self.model.state_dict(), f'Trained_model/NCBF/NCBF_Obs{epoch}.pt'.format(epoch))
            torch.save(self.model.state_dict(), f'Trained_model/NCBF/NCBF_Obs{self.run}.pt'.format(self.run))



# sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
# fault_list = FaultPattern(sensor_list,
#                           fault_target=[[1], [2, 3]],
#                           fault_value=[[0.1], [0.15, 2]])
# ObsAvoid = ObsAvoid()
# gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01]
# CR = Conflict_Resolution()