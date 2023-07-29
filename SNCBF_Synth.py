import torch
from NCBF_Synth import NCBF_Synth
from tqdm import tqdm
# from progress.bar import Bar
from Modules.NCBF import *
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Cases.ObsAvoid import ObsAvoid
from Verifier.SVerifier import Stochastic_Verifier
# from Critic_Synth.NCritic import *
import time
from EKF import *
# from collections import OrderedDict

class SNCBF_Synth(NCBF_Synth):
    '''
    Synthesize an NCBF for a given safe region h(x)
    for given a system with polynomial f(x) and g(x)
    '''
    def __init__(self, arch, act_layer, case, verbose=False):
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
        lctime = time.ctime(time.time())
        # Tensorboard
        self.writer = SummaryWriter(f'./runs/SNCBF/{lctime}'.format(lctime))

        # Initialize stochastic term related data
        self.gamma = 0.1
        self.delta_gamma = torch.zeros(1)
        self.c = torch.diag(torch.ones(self.DIM))
        self.ekf_gain = torch.Tensor([[0.06415174, -0.01436932, -0.04649317],
                                      [-0.06717124, 0.02750288,  0.14107035],
                                      [-0.0201735,  0.00625575, -0.0836058]])
        # [[0.06415174 -0.01436932 -0.04649317]
        #  [-0.06717124 0.02750288  0.14107035]
        #  [-0.0201735  0.00625575 -0.0836058]]
        self.run = 0
        # Verifier proposed to verify feasibility
        self.veri = Stochastic_Verifier(NCBF=self, case=case,
                                        EKFGain=self.ekf_gain,
                                        grid_shape=[100, 100, 100],
                                        verbose=verbose)


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
        vec_gamma = torch.amax(torch.abs(grad), 1)
        delta_gamma = torch.norm(vec_gamma) * gamma
        # return torch.max(delta_gamma, self.delta_gamma)
        return delta_gamma

    # def EKF(self):
    #     landmarks = np.array([[5, 10, 0.5], [10, 5, 0.5], [15, 15, 0.5]])
    #
    #     ekf = run_localization(
    #         landmarks, std_vel=0.1, std_steer=np.radians(1),
    #         std_range=0.3, std_bearing=0.1)
    #     print('Final P:', ekf.P.diagonal())
    #     return ekf.K

    def feasibility_loss(self, grad_vector, X_batch):
        # compute loss based on (db/dx)*fx + (db/dx)*gx*u
        dbdxfx = (grad_vector.transpose(0, 1).unsqueeze(1)
                  @ self.case.f_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        dbdxgx = (grad_vector.transpose(0, 1).unsqueeze(1)
                  @ self.case.g_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        u = self.feasible_u(dbdxfx, dbdxgx)
        # update delta_gamma
        self.delta_gamma = self.numerical_delta_gamma(grad_vector, self.gamma)
        EKF_term = grad_vector.transpose(0,1) @ self.ekf_gain @ self.c
        stochastic_term = -self.gamma * EKF_term.norm(dim=1)
        feasibility_output = dbdxfx + dbdxgx * u + stochastic_term
        return feasibility_output

    def feasible_violations(self, model_output, feasibility_output, batch_length, rlambda):
        # b_gamma = (model_output - self.delta_gamma)
        b_gamma = model_output
        violations = -1 * feasibility_output - rlambda * torch.abs(b_gamma.transpose(0, 1))
        # return violations
        return torch.max(violations, torch.zeros([1, batch_length]))

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
        size = 80
        rdm_input = self.generate_data(size)
        # rdm_input = self.generate_input(shape)
        # ref_output = torch.unsqueeze(self.h_x(rdm_input.transpose(0, self.DIM)), self.DIM)
        ref_output = self.case.h_x(rdm_input).unsqueeze(1)
        normalized_ref_output = torch.tanh(10*ref_output)
        batch_length = 16**self.DIM
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
                if feasibility_running_loss <= 0.0001 and not warm_start:
                    try:
                        veri_result, num = self.veri.proceed_verification()
                    except:
                        pass


            pbar.close()
            if feasibility_running_loss <= 0.0001 and not warm_start:
                try:
                    veri_result, num = self.veri.proceed_verification()
                except:
                    pass
            # if veri_result:
            #     torch.save(self.model.state_dict(), f'Trained_model/NCBF/NCBF_Obs{epoch}.pt'.format(epoch))
            torch.save(self.model.state_dict(), f'Trained_model/NCBF/SNCBF_Obs{self.run}.pt'.format(self.run))

# ObsAvoid = ObsAvoid()
# newCBF = SNCBF_Synth([32, 32], [True, True], ObsAvoid, verbose=True)
# # newCBF.train(50, warm_start=True)
# # newCBF.run += 1
# newCBF.train(num_epoch=10, num_restart=8, warm_start=False)
# # newCBF.model.load_state_dict(torch.load('Trained_model/SNCBF/SNCBFGood/SNCBF_Obs0.pt'))
#
# # There is a bug in verifier that causes memory error due to too many intersections to verify
# veri_result, num = newCBF.veri.proceed_verification()
