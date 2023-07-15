import torch

from Modules.NCBF import *
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Cases.ObsAvoid import ObsAvoid
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
        self.veri = Verifier(NCBF=self, case=case, grid_shape=[20, 20, 20], verbose=verbose)

    def numerical_gradient(self, X_batch, model_output, batch_length, epsilon=0.001):
        grad = []
        for i in range(self.DIM):
            gradStep = torch.zeros(self.DIM)
            gradStep[i] += epsilon
            gradData = X_batch + gradStep
            dbdxi = ((self.forward(gradData) - model_output) / epsilon).reshape([batch_length])
            grad.append(dbdxi)

        return grad

    def feasible_con(self, u, dbdxfx, dbdxgx):
        return np.min(dbdxfx + dbdxgx * u)

    def feasible_u(self, dbdxfx, dbdxgx, min_flag=False):
        if min_flag:
            df = dbdxfx.detach().numpy()
            dg = dbdxgx.detach().numpy()
            res_list = []
            for i in range(len(df)):
                res = minimize(self.feasible_con, x0=np.zeros(1), args=(df[i], dg[i]))
                # pos_res = np.max([res.fun, np.zeros(len([res.fun]))])
                res_list.append(res.x)
            return torch.Tensor(res_list).squeeze()
        else:
            [u_lb, u_ub] = torch.Tensor(self.case.CTRLDOM)
            res_list = []
            for i in range(len(dbdxfx)):
                res_ulb = dbdxfx[i] + dbdxgx[i] * u_lb
                res_uub = dbdxfx[i] + dbdxgx[i] * u_ub
                res = torch.max(res_ulb, res_uub)
                res_list.append(res)
            return torch.Tensor(res_list).squeeze()

    def feasibility_loss(self, grad_vector, X_batch):
        dbdxfx = (grad_vector.transpose(0, 1).unsqueeze(1)
                  @ self.case.f_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        dbdxgx = (grad_vector.transpose(0, 1).unsqueeze(1)
                  @ self.case.g_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        u = self.feasible_u(dbdxfx, dbdxgx)
        feasibility_output = dbdxfx + dbdxgx * u
        return feasibility_output

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
        optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        # Generate data
        size = 100
        shape = []
        for _ in range(self.DIM):
            shape.append(size)
        rlambda = 1
        rdm_input = self.generate_data(size)
        # rdm_input = self.generate_input(shape)
        # ref_output = torch.unsqueeze(self.h_x(rdm_input.transpose(0, self.DIM)), self.DIM)
        ref_output = self.h_x(rdm_input.transpose(0, 1)).unsqueeze(1)
        batch_length = 2**self.DIM
        training_loader = DataLoader(list(zip(rdm_input, ref_output)), batch_size=batch_length, shuffle=True)

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
                # dbdxfx = (grad_vector.transpose(0, 1).unsqueeze(1)
                #           @ self.case.f_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
                # dbdxgx = (grad_vector.transpose(0, 1).unsqueeze(1)
                #           @ self.case.g_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
                feasibility_output = self.feasibility_loss(grad_vector, X_batch)
                check_item = torch.max((-torch.abs(model_output)+0.1).reshape([1, batch_length]), torch.zeros([1, batch_length]))
                # feasibility_loss = torch.sum(torch.tanh(check_item*feasibility_output))

                # Our loss function
                # violations = -check_item * feasibility_output
                # Chuchu Fan loss function
                violations = -1 * feasibility_output - torch.max(rlambda * torch.abs(model_output.transpose(0, 1)),
                                                                 torch.zeros([1, batch_length]))
                feasibility_loss = 100 * torch.sum(torch.max(violations - 1e-4, torch.zeros([1, batch_length])))
                loss = self.def_loss(1*correctness_loss + 1*feasibility_loss + 1*trivial_loss)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                feasibility_running_loss += feasibility_loss.item()
                correctness_running_loss += correctness_loss.item()
                trivial_running_loss += trivial_loss.item()
                # if epoch % 50 == 49:
                #     print('[%d] loss: %.3f' % (epoch + 1, running_loss / 2000))
            # if epoch % 25 == 24:
                print('[%d] loss: %.3f' % (epoch + 1, running_loss))
                print('[%d] Floss: %.3f' % (epoch + 1, feasibility_running_loss))
                print('[%d] Closs: %.3f' % (epoch + 1, correctness_running_loss))
                print('[%d] Tloss: %.3f' % (epoch + 1, trivial_running_loss))

            if epoch % 100 == 99:
                visualize(self.model)
                scheduler.step()
            if epoch % 200 == 199:
                veri_result, num = self.veri.proceed_verification()
                print(veri_result)



# Define Case
# x0, x1 = sp.symbols('x0, x1')

ObsAvoid = ObsAvoid()
newCBF = NCBF_Synth([32, 32], [True, True], ObsAvoid, verbose=False)
# newCBF.veri.proceed_verification()
for restart in range(3):
    newCBF.train(1000)

visualize(newCBF)