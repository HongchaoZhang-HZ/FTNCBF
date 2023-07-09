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

    def safe_correctness(self, ref_output, model_output, l_co=1):
        norm_model_output = torch.tanh(model_output)
        length = len(-ref_output + norm_model_output)
        # norm_ref_output = torch.tanh(ref_output)
        FalsePositive_loss = torch.max(-ref_output.reshape([1, length]), torch.zeros([1, length])) * \
                             torch.max((model_output + 0.01).reshape([1, length]), torch.zeros([1, length]))
        FalseNegative_loss = torch.max(ref_output.reshape([1, length]), torch.zeros([1, length])) * \
                             torch.max((-model_output + 0.01).reshape([1, length]), torch.zeros([1, length]))
        loss = l_co * torch.sum(FalsePositive_loss + 0.001*FalseNegative_loss)
        return loss

    def train(self, num_epoch):
        optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
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
            for X_batch, y_batch in training_loader:

                optimizer.zero_grad()
                model_output = self.forward(X_batch)

                warm_start_loss = self.warm_start(y_batch, model_output)
                correctness_loss = self.safe_correctness(y_batch, model_output, 1)
                # x[1] + 2 * x[0] * x[1], -x[0] + 2 * x[0] ** 2 - x[1] ** 2
                dx0data = X_batch + torch.Tensor([0.001, 0])
                dx1data = X_batch + torch.Tensor([0, 0.001])
                dbdx0 = ((self.forward(dx0data) - model_output)/0.001).reshape([batch_length])
                dbdx1 = ((self.forward(dx1data) - model_output)/0.001).reshape([batch_length])
                feasibility_output = dbdx0 * (X_batch[:,0] + 2*X_batch[:,0]*X_batch[:,1]) \
                                     + dbdx1 * (-X_batch[:,0] + 2*X_batch[:,0]**2 - X_batch[:,1]**2)
                check_item = torch.max((-torch.abs(model_output)+0.2).reshape([1, batch_length]), torch.zeros([1, batch_length]))
                # feasibility_loss = torch.sum(torch.tanh(check_item*feasibility_output))
                violations = -check_item * feasibility_output
                feasibility_loss = torch.sum(torch.max(violations-1e-4*torch.ones([1,batch_length]), torch.zeros([1, batch_length])))
                # feasibility_loss = torch.sum(torch.tanh(-feasibility_output))
                # feasibility_loss = torch.sum(torch.max(feasibility_output, torch.zeros([1, batch_length])))
                loss = self.def_loss(correctness_loss + feasibility_loss)

                loss.backward()
                optimizer.step()


                running_loss += loss.item()
                # if epoch % 50 == 49:
                #     print('[%d] loss: %.3f' % (epoch + 1, running_loss / 2000))

            if epoch % 50 == 49:
                print('[%d] loss: %.3f' % (epoch + 1, loss))
                running_loss = 0.0
                veri_result, num = self.veri.proceed_verification()
                print(veri_result)
                visualize(self.model)
                scheduler.step()


# Define Case
# x0, x1 = sp.symbols('x0, x1')

hx = lambda x: (x[0] + x[1] ** 2)
# hx = (x0 + x1**2)
# x0dot = x1 + 2*x0*x1
# x1dot = -x0 + 2*x0**2 - x1**2
fx = lambda x: [x[1] + 2*x[0]*x[1], -x[0] + 2*x[0]**2 - x[1]**2]
gx = [0, 0]
Darboux = case(fx, gx, hx, [[-2,2],[-2,2]], [])
newCBF = NCBF_Synth([10, 10], [True, True], [[-2, 2], [-2, 2]], Darboux, verbose=True)
# newCBF.model.load_state_dict(torch.load('darboux_2_10.pt'))
# new_state_dict = OrderedDict()
# new_state_dict = newCBF.model.state_dict()
# new_state_dict['4.weight'] = -new_state_dict['4.weight']
# new_state_dict['4.bias'] = -new_state_dict['4.bias']
# testCBF = NCBF_Synth([10, 10], [True, True], [[-2, 2], [-2, 2]], Darboux, verbose=True)
# testCBF.model.load_state_dict(new_state_dict)
# veri_result, num = testCBF.veri.proceed_verification()
# newCBF.model.load_state_dict(testCBF.model.state_dict())
newCBF.train(1000)
# res = testCritic.PolyVerification(newCB

visualize(newCBF)