import numpy as np
import torch
from SensorFaults import *
from NCBF_Synth import NCBF_Synth
from tqdm import tqdm
import itertools
# from progress.bar import Bar
from Modules.NCBF import *
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.autograd.functional import hessian
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Cases.ObsAvoid import ObsAvoid
from Verifier.SVerifier import Stochastic_Verifier
# from Critic_Synth.NCritic import *
import time
from EKF import *
from FTEst import *
# from collections import OrderedDict

class SNCBF_Synth(NCBF_Synth):
    '''
    Synthesize an NCBF for a given safe region h(x)
    for given a system with polynomial f(x) and g(x)
    '''
    def __init__(self, arch, act_layer, case,
                 sensors: SensorSet,
                 fault_target: list,
                 fault_value: list,
                 sigma, nu, gamma_list,
                 verbose=False):
        '''
        Input architecture and ReLU layers, input case, verbose for display
        :param arch: [list of int] architecture of the NN
        :param act_layer: [list of bool] if the corresponding layer with ReLU, then True
        :param case: Pre-defined case class, with f(x), g(x) and h(x)
        :param verbose: Flag for display or not
        '''
        super().__init__(arch, act_layer, case, verbose=False)
        # Under construction: Critic is designed to tuning loss fcn automatically
        # self.critic = NeuralCritic(case)
        # lctime = time.ctime(time.time())
        lctime = time.strftime("%Y%m%d%H%M%S")
        # Tensorboard
        self.writer = SummaryWriter(f'./runs/SNCBF/{lctime}'.format(lctime))

        # Initialize stochastic term related data
        self.gamma = 0.1
        self.sigma = torch.tensor(sigma, dtype=torch.float).unsqueeze(1)
        self.nu = torch.tensor(nu, dtype=torch.float).unsqueeze(1)
        self.delta_gamma = torch.zeros(1)
        self.c = torch.diag(torch.ones(self.DIM))
        self.ekf_gain = torch.Tensor([[0.06415174, -0.01436932, -0.04649317],
                                      [-0.06717124, 0.02750288,  0.14107035],
                                      [-0.0201735,  0.00625575, -0.0836058]])
        # [[0.06415174 -0.01436932 -0.04649317]
        #  [-0.06717124 0.02750288  0.14107035]
        #  [-0.0201735  0.00625575 -0.0836058]]
        self.gamma = gamma_list
        self.run = 0
        # Verifier proposed to verify feasibility
        self.veri = Stochastic_Verifier(NCBF=self, case=case,
                                        EKFGain=self.ekf_gain,
                                        grid_shape=[100, 100, 100],
                                        verbose=verbose)

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
        self.FTEst = FTEst(None, self.sensor_list, self.fault_list)
        self.FTEKF_gain_list = self.FTEst.EKFgain_list

    @property
    def get_model(self):
        return self.model

    # def get_grad(self, x):
    #     grad_input = torch.tensor(x, requires_grad=True, dtype=torch.float)
    #     return torch.autograd.grad(self.model.forward(grad_input), grad_input)

    def get_hessian(self, x):
        grad_input = torch.tensor(x, requires_grad=True, dtype=torch.float)
        hessian_matrix = hessian(self.model.forward, grad_input).squeeze()
        return hessian_matrix

    def update_EKF_gain(self, new_gain):
        self.ekf_gain = new_gain

    def update_obs_matrix_c(self, obs_matrix):
        self.c = obs_matrix

    def numerical_delta_gamma(self, grad, gamma):
        '''
        We can numerically get b_gamma by decrease along b, i.e. delta gamma
        To compute the delta gamma, we get the largest gradient
        :param grad: [torch.Tensor] vector of numerical gradient
        :param gamma: [scalar] gamma of state x
        :return: delta gamma
        '''
        vec_gamma = torch.amax(torch.abs(grad), 0)
        delta_gamma = torch.norm(vec_gamma) * gamma
        # return torch.max(delta_gamma, self.delta_gamma)
        return delta_gamma

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

    def feasibility_loss(self, grad_vector, X_batch, ekf_gain, gamma):
        # compute loss based on (db/dx)*fx + (db/dx)*gx*u
        # dbdxfx = (grad_vector.transpose(0, 1).unsqueeze(1)
        #           @ self.case.f_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        # dbdxgx = (grad_vector.transpose(0, 1).unsqueeze(1)
        #           @ self.case.g_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        # compute loss based on (db/dx)*fx + (db/dx)*gx*u
        dbdxfx = (grad_vector.unsqueeze(-1).transpose(-1, -2)
                  @ self.case.f_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        dbdxgx = (grad_vector.unsqueeze(-1).transpose(-1, -2)
                  @ self.case.g_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        u = self.feasible_u(dbdxfx, dbdxgx)
        # u = -X_batch[:,-1]
        # update delta_gamma
        self.delta_gamma = self.numerical_delta_gamma(grad_vector, gamma)
        # EKF_term = grad_vector.transpose(0,1) @ self.ekf_gain @ self.c
        EKF_term = grad_vector @ self.ekf_gain @ self.c
        stochastic_term = -gamma * EKF_term.norm(dim=1)

        # second order term
        hes_array = []
        trace_term_list = []
        for idx in range(len(X_batch)):
            hess = self.get_hessian(X_batch[idx])
            second_order_term = self.nu.transpose(0, 1).numpy() @ ekf_gain.transpose() \
                                @ hess.numpy() @ ekf_gain @ self.nu.numpy()
            trace_term = second_order_term.trace()
            trace_term_list.append(trace_term)
        trace_term_list = torch.tensor(np.array(trace_term_list))

        feasibility_output = dbdxfx + dbdxgx * u + stochastic_term + trace_term_list
        return feasibility_output

    def feasible_violations(self, model_output, feasibility_output, batch_length, rlambda):
        # b_gamma = (model_output - self.delta_gamma)
        b_gamma = model_output
        violations = -1 * feasibility_output - rlambda * torch.abs(b_gamma.transpose(0, 1))
        # return violations
        return torch.max(violations, torch.zeros([1, batch_length]))

    def train(self, num_epoch, num_restart=10, alpha1=None, alpha2=None, warm_start=False):
        if warm_start:
            learning_rate = 1e-2
        else:
            learning_rate = 1e-5
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        # define hyper-parameters
        alphaf, alpha1, alpha2 = 1, 1, 0.0001
        # 1, 1e-8
        # Set alpha2=0 for feasibility test with Floss quickly converge to 0
        # If set alpha2 converges but does not pass the verification, then increase the sampling number.
        # This problem is caused by lack of counter examples and can be solved by introducing CE from Verifier
        rlambda = 1

        # Generate data
        size = 32
        volume = torch.Tensor([0])
        volume_pre = torch.Tensor([0])
        rdm_input = self.generate_data(size)
        # rdm_input = self.generate_input(shape)
        # ref_output = torch.unsqueeze(self.h_x(rdm_input.transpose(0, self.DIM)), self.DIM)
        ref_output = self.case.h_x(rdm_input).unsqueeze(1)
        normalized_ref_output = torch.tanh(10*ref_output)
        # batch_length = 16**self.DIM
        batch_length = 32
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
                    correctness_loss, coverage_loss = self.safe_correctness(y_batch, model_output, l_co=1,
                                                                            alpha1=alpha1, alpha2=alpha2)
                    # trivial_loss = self.trivial_panelty(ref_output, self.model.forward(rdm_input), 1)
                    trivial_loss = self.trivial_panelty(y_batch, model_output, 1)

                    # grad = self.numerical_gradient(X_batch, model_output, batch_length, epsilon=0.001)
                    # grad_vector = torch.vstack(grad)
                    feasibility_output = 0
                    ce_indicator = 0
                    grad_vector = torch.vstack([self.get_grad(x)[0] for x in X_batch])
                    for i in range(len(self.FTEKF_gain_list)):
                        tempt_feas_loss = self.feasibility_loss(grad_vector, X_batch,
                                                                self.FTEKF_gain_list[i], self.gamma[i])
                        feasibility_output += batch_length * (torch.relu(model_output.squeeze()) *
                                                              torch.relu(-model_output.squeeze()+1e-2) *
                                                              (tempt_feas_loss + model_output.squeeze()))
                        ce_indicator += (torch.relu(model_output.squeeze())/model_output.squeeze() *
                                        torch.relu(-model_output.squeeze()+1e-1)/(-model_output.squeeze()+1e-1) *
                                        torch.relu(-tempt_feas_loss - model_output.squeeze())
                                        /((-tempt_feas_loss - model_output.squeeze())))
                    # check_item = torch.max((-torch.abs(model_output)+0.2).reshape([1, batch_length]), torch.zeros([1, batch_length]))
                    # feasibility_loss = torch.sum(torch.tanh(check_item*feasibility_output))

                    # Our loss function
                    # violations = -check_item * self.feasible_violations(model_output, feasibility_output, batch_length, rlambda)
                    # Chuchu Fan loss function
                    # violations = -1 * self.feasible_violations(model_output, feasibility_output, batch_length, rlambda)
                    # violations = -1 * feasibility_output - torch.max(rlambda * torch.abs(model_output.transpose(0, 1)),
                    #                                                  torch.zeros([1, batch_length]))
                    # feasibility_loss = 2 * torch.sum(torch.max(violations - 1e-4, torch.zeros([1, batch_length])))
                    feasibility_loss = torch.relu(-feasibility_output).sum()
                    mseloss = torch.nn.MSELoss()
                    # loss = self.def_loss(1 * correctness_loss + 1 * feasibility_loss + 1 * trivial_loss)
                    # floss = mseloss(torch.max(violations - 1e-4, torch.zeros([1, batch_length])), torch.zeros(batch_length))
                    # tloss = mseloss(trivial_loss, torch.Tensor([0.0]))
                    if warm_start:
                        correctness_loss, coverage_loss = self.safe_correctness(y_batch, model_output, l_co=1, alpha1=1,
                                                                                alpha2=0.0001)
                        loss = correctness_loss + coverage_loss + trivial_loss
                    else:
                        # loss = feasibility_loss
                        loss = (alpha1 * correctness_loss + alpha2 * coverage_loss
                                + alphaf * feasibility_loss + trivial_loss)
                    loss.backward()
                    # with torch.no_grad():
                    #     loss = torch.max(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    # alpha1 += 0.1 * correctness_loss.item()
                    alpha1 += 0.001 * correctness_loss.item()
                    alphaf += 100 * feasibility_loss.item()
                    # alphaf += 0.1 * feasibility_loss.item()
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
                # if not warm_start:
                volume = self.compute_volume(rdm_input)
                self.writer.add_scalar('Loss/Loss', running_loss, self.run * num_epoch + epoch)
                self.writer.add_scalar('Loss/FLoss', feasibility_running_loss.item(), self.run * num_epoch + epoch)
                self.writer.add_scalar('Loss/CLoss', correctness_running_loss.item(), self.run * num_epoch + epoch)
                self.writer.add_scalar('Loss/TLoss', trivial_running_loss.item(), self.run * num_epoch + epoch)
                pbar.set_postfix({'Loss': running_loss,
                                  'Floss': feasibility_running_loss.item(),
                                  'Closs': correctness_running_loss.item(),
                                  'Tloss': trivial_running_loss.item(),
                                  'WarmUp': str(warm_start),
                                  'FCEnum': ce_indicator.sum().item(),
                                  'Vol': volume.item()})
                pbar.update(1)
                scheduler.step()
                # Log volume of safe region

                # self.writer.add_scalar('Volume', volume, self.run * num_epoch + epoch)
                # scheduler.step()
                if correctness_running_loss.item() < 1 and trivial_running_loss.item() < 0.001 and ce_indicator.sum().item() == 0 and volume > 0.2:
                    if volume_pre < volume:
                        torch.save(self.model.state_dict(),
                                   f'Trained_model/NCBF/SNCBF_SoloGood{self.run}.pt'.format(self.run))
                        volume_pre = volume
                # if feasibility_running_loss <= 0.0001 and not warm_start:
                #     try:
                #         veri_result, num = self.veri.proceed_verification()
                #     except:
                #         pass
            pbar.close()
            # if feasibility_running_loss <= 0.0001 and not warm_start:
            #     try:
            #         veri_result, num = self.veri.proceed_verification()
            #     except:
            #         pass
            # if veri_result:
            #     torch.save(self.model.state_dict(), f'Trained_model/NCBF/NCBF_Obs{epoch}.pt'.format(epoch))
            # torch.save(self.model.state_dict(), f'Trained_model/SNCBF/SNCBF_Obs{self.run}.pt'.format(self.run))

ObsAvoid = ObsAvoid()
sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
fault_list = FaultPattern(sensor_list,
                          fault_target=[[1], [2]],
                          fault_value=[[0.1], [0.15]])
gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01]
newCBF = SNCBF_Synth([32, 32], [True, True], ObsAvoid,
                     sensors=sensor_list,
                     fault_target=[{1}, {2}],
                     fault_value=[[0.1], [0.15]],
                     sigma=[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
                     nu=[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
                     gamma_list=gamma_list, verbose=True)
# newCBF.train(50, warm_start=True)
# newCBF.run += 1
# newCBF.model.load_state_dict(torch.load('Trained_model/SNCBF/SNCBFGood/SNCBF_Obs0.pt'))
newCBF.train(num_epoch=20, num_restart=5, warm_start=False)
# newCBF.model.load_state_dict(torch.load('Trained_model/SNCBF/SNCBFGood/SNCBF_Obs0.pt'))
#
# # There is a bug in verifier that causes memory error due to too many intersections to verify
# veri_result, num = newCBF.veri.proceed_verification()
