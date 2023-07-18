import torch
from tqdm import tqdm
from Modules.NCBF import *
from torch import optim
from filterpy.kalman import ExtendedKalmanFilter
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Cases.Darboux import Darboux
# import cma
from cmaes import CMA
from Verifier import Verifier
from collections import OrderedDict
from Critic_Synth.NCritic import *

class NCBF_Synth(NCBF):
    def __init__(self,arch, act_layer, case, verbose=False):
        self.case = case
        DOMAIN = self.case.DOMAIN
        super().__init__(arch, act_layer, DOMAIN)
        self.critic = NeuralCritic(case)
        self.veri = Verifier(NCBF=self, case=case, grid_shape=[100, 100], verbose=verbose)

    def numerical_gradient(self, X_batch, model_output, batch_length, epsilon=0.001):
        grad = []
        for i in range(self.DIM):
            gradStep = torch.zeros(self.DIM)
            gradStep[i] += epsilon
            gradData = X_batch + gradStep
            dbdxi = ((self.forward(gradData) - model_output) / epsilon).reshape([batch_length])
            grad.append(dbdxi)
        return grad

    def numerical_b_gamma(self, grad, gamma):
        # todo: debug
        return np.max(np.abs(grad)) * gamma

    def EKF(self):
        # todo: extended kalman filter gain for different sensor failure
        K = torch.ones([self.DIM, self.DIM])
        return K

    def feasibility_loss(self, grad_vector, X_batch, model_output, batch_length, gamma=0.1):
        # todo: debug
        # todo: tuning
        rlambda = 1
        c = 1
        feasibility_output = (grad_vector.transpose(0, 1).unsqueeze(1)
                              @ self.case.f_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        check_item = torch.max((-torch.abs(model_output) + 0.1).reshape([1, batch_length]),
                               torch.zeros([1, batch_length]))
        stochastic_term = -gamma * torch.linalg.norm(grad_vector @ self.EKF() * c)

        # Our loss function
        # violations = -check_item * feasibility_output
        # Chuchu Fan loss function
        delta_gamma = self.numerical_b_gamma(grad_vector, gamma)
        violations = -1 * feasibility_output - stochastic_term\
                     - torch.max(rlambda * torch.abs((model_output-delta_gamma).transpose(0, 1)),
                                 torch.zeros([1, batch_length]))
        feasibility_loss = 100 * torch.sum(torch.max(violations - 1e-4, torch.zeros([1, batch_length])))
        return feasibility_loss

    def safe_correctness(self, ref_output, model_output, l_co=1, alpha1=1, alpha2=0.001):
        norm_model_output = torch.tanh(model_output)
        length = len(-ref_output + norm_model_output)
        # norm_ref_output = torch.tanh(ref_output)
        FalsePositive_loss = torch.max(-ref_output.reshape([1, length]), torch.zeros([1, length])) * \
                             torch.max((model_output + 0.01).reshape([1, length]), torch.zeros([1, length]))
        FalseNegative_loss = torch.max(ref_output.reshape([1, length]), torch.zeros([1, length])) * \
                             torch.max((-model_output + 0.01).reshape([1, length]), torch.zeros([1, length]))
        loss = l_co * torch.sum(alpha1*FalsePositive_loss + alpha2*FalseNegative_loss)
        return loss

    def trivial_panelty(self, ref_output, model_output, coeff=1, epsilon=0.1):
        min_ref = torch.max(ref_output)
        max_ref = torch.min(ref_output)
        # if max_ref >= 1e-4 and min_ref <= -1e-4:
        #     non_pos_loss = coeff * torch.max(0.5 - torch.max(model_output), torch.zeros(1))
        #     non_neg_loss = coeff * torch.max(0.5 - torch.max(-model_output), torch.zeros(1))
        if max_ref >= 1e-4 and min_ref >= 1e-4:
            non_pos_loss = torch.zeros(1)
            non_neg_loss = torch.zeros(1)
        elif max_ref <= -1e-4 and min_ref <= -1e-4:
            non_pos_loss = torch.zeros(1)
            non_neg_loss = torch.zeros(1)
        else:
            non_pos_loss = coeff * torch.max(epsilon - torch.max(model_output), torch.zeros(1))
            non_neg_loss = coeff * torch.max(epsilon - torch.max(-model_output), torch.zeros(1))
        loss = non_pos_loss + non_neg_loss
        return loss

    def train(self, num_epoch):
        optimizer = optim.SGD(self.model.parameters(), lr=1e-4)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        # Generate data
        size = 100
        shape = []
        for _ in range(self.DIM):
            shape.append(size)

        rdm_input = self.generate_data(size)
        # rdm_input = self.generate_input(shape)
        # ref_output = torch.unsqueeze(self.h_x(rdm_input.transpose(0, self.DIM)), self.DIM)
        ref_output = self.h_x(rdm_input.transpose(0, 1)).unsqueeze(1)
        batch_length = 16
        training_loader = DataLoader(list(zip(rdm_input, ref_output)), batch_size=batch_length, shuffle=True)
        pbar = tqdm(total=num_epoch)
        veri_result = False
        for epoch in range(num_epoch):
            running_loss = 0.0
            feasibility_running_loss = 0.0
            correctness_running_loss = 0.0
            trivial_running_loss = 0.0

            for X_batch, y_batch in training_loader:

                optimizer.zero_grad()
                model_output = self.forward(X_batch)

                warm_start_loss = self.warm_start(y_batch, model_output)
                correctness_loss = self.safe_correctness(y_batch, model_output, l_co=1, alpha1=1, alpha2=0)
                trivial_loss = self.trivial_panelty(ref_output, self.model.forward(rdm_input), 1)
                grad = self.numerical_gradient(X_batch, model_output, batch_length, epsilon=0.001)
                grad_vector = torch.vstack(grad)
                feasibility_loss = self.feasibility_loss(grad_vector, X_batch, model_output, batch_length)
                loss = self.def_loss(1*correctness_loss + 1*feasibility_loss + 1*trivial_loss)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                feasibility_running_loss += feasibility_loss.item()
                correctness_running_loss += correctness_loss.item()
                trivial_running_loss += trivial_loss.item()
                # if epoch % 50 == 49:
                #     print('[%d] loss: %.3f' % (epoch + 1, running_loss / 2000))
                # Print Detailed Loss
            running_loss += loss.item()
            feasibility_running_loss += feasibility_loss.item()
            correctness_running_loss += correctness_loss.item()
            trivial_running_loss += trivial_loss.item()
            # Process Bar Print Losses
            pbar.set_postfix({'Loss': running_loss,
                              'Floss': feasibility_running_loss,
                              'Closs': correctness_running_loss,
                              'Tloss': trivial_running_loss,
                              'PVeri': str(veri_result)})
            pbar.update(1)

            if epoch % 50 == 49:
                scheduler.step()
            if epoch % 100 == 99:
                veri_result, num = self.veri.proceed_verification()
                visualize(self.model)
                # print(veri_result)



# Define Case
# x0, x1 = sp.symbols('x0, x1')

Darboux = Darboux()
newCBF = NCBF_Synth([10, 10], [True, True], Darboux, verbose=False)
newCBF.veri.proceed_verification()
for restart in range(3):
    newCBF.train(1000)

visualize(newCBF)