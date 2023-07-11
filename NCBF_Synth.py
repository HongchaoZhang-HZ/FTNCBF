import torch

from Modules.NCBF import *
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import cma
from cmaes import CMA
from Verifier import Verifier
from collections import OrderedDict
from Critic_Synth.NCritic import *

class NCBF_Synth(NCBF):
    def __init__(self,arch, act_layer, DOMAIN, case, verbose=False):
        super().__init__(arch, act_layer, DOMAIN)
        self.critic = NeuralCritic(case)
        self.veri = Verifier(NCBF=self, case=case, grid_shape=[100, 100], verbose=verbose)


    def feasibility_loss(self, model_output, grad_condition, l_co=1):
        # when model_output is close to the boundary grad condition comes in
        violations = grad_condition.reshape(torch.sigmoid(model_output).shape) * torch.sigmoid(model_output)
        # loss = torch.sum(torch.sigmoid(-violations).reshape([1, 10000]))
        violations = torch.max((-violations).reshape([1, 10000]), torch.zeros([1, 10000]))
        loss = l_co * torch.sum(violations)
        return loss

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
        shape = [100, 100]
        rlambda = 1
        vx, vy, rdm_input = self.generate_input(shape)

        ref_output = self.h_x(rdm_input.transpose(0, 1)).reshape([shape[0] * shape[1], 1])
        batch_length = 16
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((1,), (1,))])
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
                # x[1] + 2 * x[0] * x[1], -x[0] + 2 * x[0] ** 2 - x[1] ** 2
                dx0data = X_batch + torch.Tensor([0.001, 0])
                dx1data = X_batch + torch.Tensor([0, 0.001])
                dbdx0 = ((self.forward(dx0data) - model_output)/0.001).reshape([batch_length])
                dbdx1 = ((self.forward(dx1data) - model_output)/0.001).reshape([batch_length])
                feasibility_output = dbdx0 * (X_batch[:,0] + 2*X_batch[:,0]*X_batch[:,1]) \
                                     + dbdx1 * (-X_batch[:,0] + 2*X_batch[:,0]**2 - X_batch[:,1]**2)
                check_item = torch.max((-torch.abs(model_output)+0.1).reshape([1, batch_length]), torch.zeros([1, batch_length]))
                # feasibility_loss = torch.sum(torch.tanh(check_item*feasibility_output))
                violations = -check_item * feasibility_output
                # violations = -1 * feasibility_output - torch.max(rlambda * model_output.transpose(0, 1), torch.zeros([1, batch_length]))
                feasibility_loss = 100*torch.sum(torch.max(violations-1e-4, torch.zeros([1, batch_length])))
                # feasibility_loss = torch.sum(torch.tanh(-feasibility_output))
                # feasibility_loss = torch.sum(torch.max(feasibility_output, torch.zeros([1, batch_length])))
                loss = self.def_loss(1*correctness_loss + 1*feasibility_loss + 1*trivial_loss)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                feasibility_running_loss += feasibility_loss.item()
                correctness_running_loss += correctness_loss.item()
                trivial_running_loss += trivial_loss.item()
                # if epoch % 50 == 49:
                #     print('[%d] loss: %.3f' % (epoch + 1, running_loss / 2000))
            if epoch % 25 == 24:
                print('[%d] loss: %.3f' % (epoch + 1, running_loss))
                print('[%d] Floss: %.3f' % (epoch + 1, feasibility_running_loss))
                print('[%d] Closs: %.3f' % (epoch + 1, correctness_running_loss))
                print('[%d] Tloss: %.3f' % (epoch + 1, trivial_running_loss))
                running_loss = 0.0
            if epoch % 100 == 99:
                visualize(self.model)
                scheduler.step()
            if epoch % 200 == 199:
                veri_result, num = self.veri.proceed_verification()
                print(veri_result)



# Define Case
# x0, x1 = sp.symbols('x0, x1')

hx = lambda x: (x[0] + x[1] ** 2)
# hx = (x0 + x1**2)
# x0dot = x1 + 2*x0*x1
# x1dot = -x0 + 2*x0**2 - x1**2
fx = lambda x: [x[1] + 2*x[0]*x[1], -x[0] + 2*x[0]**2 - x[1]**2]
gx = [0, 0]
Darboux = case(fx, gx, hx, [[-2,2],[-2,2]], [])
newCBF = NCBF_Synth([10, 10], [True, True], [[-2, 2], [-2, 2]], Darboux, verbose=False)
newCBF.model.load_state_dict(torch.load('darboux_2_10.pt'))
new_state_dict = OrderedDict()
new_state_dict = newCBF.model.state_dict()
new_state_dict['4.weight'] = -new_state_dict['4.weight']
new_state_dict['4.bias'] = -new_state_dict['4.bias']
testCBF = NCBF_Synth([10, 10], [True, True], [[-2, 2], [-2, 2]], Darboux, verbose=True)
testCBF.model.load_state_dict(new_state_dict)
veri_result, num = testCBF.veri.proceed_verification()
newCBF.model.load_state_dict(testCBF.model.state_dict())
for restart in range(3):
    newCBF.train(1000)

visualize(newCBF)