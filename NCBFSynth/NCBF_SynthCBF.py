import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
import torch
from tqdm import tqdm
from scipy.optimize import minimize
# from progress.bar import Bar
from Modules.NCBF import *
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Cases.ObsAvoid import ObsAvoid
# from Verifier.Verifier import Verifier
# from Critic_Synth.NCritic import *
import time
from Visualization.visualization import visualize
# from collections import OrderedDict

class NCBF_Synth(NCBF):
    '''
    Synthesize an NCBF for a given safe region h(x)
    for given a system with polynomial f(x) and g(x)
    '''
    def __init__(self,arch, act_layer, case, verbose=False):
        '''
        Input architecture and ReLU layers, input case, verbose for display
        :param arch: [list of int] architecture of the NN
        :param act_layer: [list of bool] if the corresponding layer with ReLU, then True
        :param case: Pre-defined case class, with f(x), g(x) and h(x)
        :param verbose: Flag for display or not
        '''
        self.case = case
        DOMAIN = self.case.DOMAIN
        super().__init__(arch, act_layer, DOMAIN)
        # Under construction: Critic is designed to tuning loss fcn automatically
        # self.critic = NeuralCritic(case)
        # Verifier proposed to verify feasibility
        self.veri = Verifier(NCBF=self, case=case, grid_shape=[100, 100, 100], verbose=verbose)
        # lctime = time.ctime(time.time())
        lctime = time.strftime("%Y%m%d%H%M%S")
        # Tensorboard
        self.writer = SummaryWriter(f'./runs/NCBF/{lctime}'.format(lctime))
        self.run = 0

    # def numerical_gradient(self, X_batch, model_output, batch_length, epsilon=0.001):
    #     # compute numerical gradient for each dimension by (x+dx)/dx
    #     grad = []
    #     for i in range(self.DIM):
    #         gradStep = torch.zeros(self.DIM)
    #         gradStep[i] += epsilon
    #         gradData = X_batch + gradStep
    #         dbdxi = ((self.forward(gradData) - model_output) / epsilon).reshape([batch_length])
    #         grad.append(dbdxi)
    #
    #     return grad

    def get_grad(self, x):
        grad_input = torch.tensor(x, requires_grad=True, dtype=torch.float)
        return torch.autograd.grad(self.model.forward(grad_input), grad_input)

    def feasible_con(self, u, dbdxfx, dbdxgx):
        # function to maximize: (db/dx)*fx + (db/dx)*gx*u
        return dbdxfx + dbdxgx * u

    def feasible_u(self, dbdxfx, dbdxgx, min_flag=False):
        # find u that minimize (db/dx)*fx + (db/dx)*gx*u
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
            # Quick check for scalar u only
            [[u_lb, u_ub]] = torch.Tensor(self.case.CTRLDOM)
            res_list = []
            for i in range(len(dbdxfx)):
                res_ulb = dbdxfx[i] + dbdxgx[i] * u_lb
                res_uub = dbdxfx[i] + dbdxgx[i] * u_ub
                # res = torch.max(res_ulb, res_uub)
                res = [u_lb, u_ub][np.argmax([res_ulb, res_uub])]
                res_list.append(res)
            return torch.Tensor(res_list).squeeze()

    def feasibility_loss(self, grad_vector, X_batch):
        # compute loss based on (db/dx)*fx + (db/dx)*gx*u
        dbdxfx = (grad_vector.unsqueeze(-1).transpose(-1, -2)
                  @ self.case.f_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        dbdxgx = (grad_vector.unsqueeze(-1).transpose(-1, -2)
                  @ self.case.g_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        u = self.feasible_u(dbdxfx, dbdxgx)
        # u = -X_batch[:,-1]
        feasibility_output = dbdxfx + dbdxgx * u
        # feasibility_output = dbdxfx
        return feasibility_output

    def feasible_violations(self, model_output, feasibility_output, batch_length, rlambda):
        # violations = -1 * feasibility_output - rlambda * torch.abs(model_output.transpose(0, 1))
        violations = -1 * feasibility_output - rlambda * model_output.squeeze()
        # return torch.max(violations, torch.zeros([1,batch_length]))
        return violations

    def safe_correctness(self, ref_output, model_output,
                         l_co: float = 1, alpha1: float = 1,
                         alpha2: float = 0.001) -> torch.Tensor:
        '''
        Penalize the incorrectness based on the h(x).
            If h(x) < 0 meaning the state x is unsafe, b(x) has to be negative.
                Therefore, alpha1, the gain of the penalty, can be large.
            If h(x) > 0 meaning the state is temporarily safe, b(x) can be +/-.
                To maximize coverage of b(x), a small penalty alpha2 is applied.
        :param ref_output: output of h(x)
        :param model_output: output of NN(x)
        :param l_co: gain of the loss
        :param alpha1: penalty for unsafe incorrectness
        :param alpha2: penalty for coverage
        :return: safety oriented correctness loss
        '''
        norm_model_output = torch.tanh(model_output)
        length = len(-ref_output + norm_model_output)
        # norm_ref_output = torch.tanh(ref_output)
        # FalsePositive_loss = torch.max(-ref_output.reshape([1, length]), torch.zeros([1, length])) * \
        #                      torch.max((model_output + 0.01).reshape([1, length]), torch.zeros([1, length]))
        # FalseNegative_loss = torch.max(ref_output.reshape([1, length]), torch.zeros([1, length])) * \
        #                      torch.max((-model_output + 0.01).reshape([1, length]), torch.zeros([1, length]))
        FalsePositive_loss = torch.relu(-ref_output) * torch.relu((model_output))
        FalseNegative_loss = torch.relu(ref_output) * torch.relu((-model_output))
        # loss = l_co * torch.sum(alpha1*FalsePositive_loss + alpha2*FalseNegative_loss)
        # return loss
        return torch.sum(FalsePositive_loss), torch.sum(FalseNegative_loss)

    def trivial_panelty(self, ref_output, model_output, coeff=1, epsilon=0.001):
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
            non_neg_loss = coeff * torch.max(-epsilon - torch.max(-model_output), torch.zeros(1))
        loss = non_pos_loss + non_neg_loss
        return loss

    def compute_volume(self, rdm_input, model_output=None):
        '''
        Compute volume covered by b(x)
        :param rdm_input: random uniform samples
        :return: numbers of samples (volume)
        '''
        # compute the positive volume contained by the NCBF
        if model_output is None:
            model_output = self.forward(rdm_input).squeeze()
        pos_output = torch.max(model_output, torch.zeros(len(rdm_input)))
        return torch.sum(pos_output > 0)/len(rdm_input)


    def train(self, num_epoch, num_restart=10, warm_start=False):
        if warm_start:
            learning_rate = 1e-2
        else:
            learning_rate = 1e-2
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
        size = 128
        volume = torch.Tensor([0])
        for self.run in range(num_restart):
            rdm_input = self.generate_data(size)
            # rdm_input = self.generate_input(shape)
            # ref_output = torch.unsqueeze(self.h_x(rdm_input.transpose(0, self.DIM)), self.DIM)
            ref_output = self.case.h_x(rdm_input).unsqueeze(1)
            normalized_ref_output = torch.tanh(10 * ref_output)
            # batch_length = 8**self.DIM
            batch_length = size ** (self.DIM-1)
            training_loader = DataLoader(list(zip(rdm_input, normalized_ref_output)), batch_size=batch_length,
                                         shuffle=True)

            pbar = tqdm(total=num_epoch)
            veri_result = False
            if not warm_start:
                pass
            for epoch in range(num_epoch):
                # Initialize loss
                running_loss = 0.0
                feasibility_running_loss = torch.Tensor([0.0])
                correctness_running_loss = torch.Tensor([0.0])
                trivial_running_loss = torch.Tensor([0.0])

                # Batch Training
                for X_batch, y_batch in training_loader:
                    model_output = self.forward(X_batch)

                    # warm_start_loss = self.warm_start(y_batch, model_output)
                    correctness_loss, coverage_loss = self.safe_correctness(y_batch, model_output, l_co=1, alpha1=alpha1, alpha2=alpha2)
                    # trivial_loss = self.trivial_panelty(ref_output, self.model.forward(rdm_input), 1)
                    trivial_loss = self.trivial_panelty(y_batch, model_output, 1)

                    # grad = self.numerical_gradient(X_batch, model_output, batch_length, epsilon=0.001)
                    # grad_vector = torch.vstack(grad)
                    grad_vector = torch.vstack([self.get_grad(x)[0] for x in X_batch])
                    # dbdx(fx + gx*u) should be >=0. If <0, a penalty will be added.
                    # feasibility_output = (torch.relu(model_output)*self.feasibility_loss(grad_vector, X_batch))
                    feasibility_output = batch_length * (torch.relu(model_output.squeeze()) * torch.relu(-model_output.squeeze()+1e-2) *
                                          (self.feasibility_loss(grad_vector, X_batch) + model_output.squeeze()))
                    ce_indicator = (torch.relu(model_output.squeeze())/model_output.squeeze() *
                                    torch.relu(-model_output.squeeze()+1e-2)/(-model_output.squeeze()+1e-2) *
                                    torch.relu(-self.feasibility_loss(grad_vector, X_batch) - model_output.squeeze())
                                    /((-self.feasibility_loss(grad_vector, X_batch) - model_output.squeeze())))
                    # check_item = torch.max((-torch.abs(model_output)+0.2).reshape([1, len(model_output)]), torch.zeros([1, len(model_output)]))
                    # feasibility_loss = torch.sum(torch.tanh(check_item*feasibility_output))

                    # Our loss function
                    # violations = -check_item * self.feasible_violations(model_output, feasibility_output, batch_length, rlambda)
                    feasibility_loss = torch.relu(-feasibility_output).sum()
                    # Chuchu Fan loss function
                    # violations = check_item * self.feasible_violations(model_output, feasibility_output, batch_length, rlambda)
                    # violations = -1 * feasibility_output - torch.max(rlambda * torch.abs(model_output.transpose(0, 1)),
                    #                                                  torch.zeros([1, batch_length]))
                    # feasibility_loss = 2 * torch.sum(torch.max(violations - 1e-4, torch.zeros([1, len(model_output)])))
                    if feasibility_loss <= 0.001:
                        pass
                    mseloss = torch.nn.MSELoss()
                    # loss = self.def_loss(1 * correctness_loss + 1 * feasibility_loss + 1 * trivial_loss)
                    # floss = mseloss(torch.max(violations - 1e-4, torch.zeros([1, batch_length])), torch.zeros(batch_length))
                    # tloss = mseloss(trivial_loss, torch.Tensor([0.0]))
                    if warm_start:
                        correctness_loss, coverage_loss = self.safe_correctness(y_batch, model_output, l_co=1, alpha1=1, alpha2=0.0001)
                        loss = correctness_loss + coverage_loss + trivial_loss
                    else:
                        # loss = feasibility_loss
                        loss = (alpha1*correctness_loss + alpha2*coverage_loss
                                + alphaf*feasibility_loss + trivial_loss)


                    loss.backward()
                    # with torch.no_grad():
                    #     loss = torch.max(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    alpha1 += 0.1 * correctness_loss.item()
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
                self.writer.add_scalar('Loss/Loss', running_loss, self.run*num_epoch+epoch)
                self.writer.add_scalar('Loss/FLoss', feasibility_running_loss.item(), self.run*num_epoch+epoch)
                self.writer.add_scalar('Loss/CLoss', correctness_running_loss.item(), self.run*num_epoch+epoch)
                self.writer.add_scalar('Loss/TLoss', trivial_running_loss.item(), self.run*num_epoch+epoch)
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
                volume = self.compute_volume(rdm_input)
                self.writer.add_scalar('Volume', volume, self.run*num_epoch+epoch)
                # self.writer.add_scalar('Verifiable', veri_result, self.run * num_epoch + epoch)
                # Process Bar Print Losses



            pbar.close()
            # visualize(self.model)
            # torch.save(self.model.state_dict(), f'Trained_model/NCBF/NCBF_Obs{self.run}.pt'.format(self.run))

ObsAvoid = ObsAvoid()

newCBF = NCBF_Synth([32, 32], [True, True], ObsAvoid, verbose=True)
# newCBF.model.load_state_dict(torch.load('WarmModel2.pt'))
# newCBF.train(num_epoch=20, num_restart=2, warm_start=True)
# layers_to_freeze = ['0.weight', '0.bias']
# for name, param in newCBF.model.named_parameters():
#     # Check if the current parameter belongs to a layer you want to freeze
#     if any(layer_name in name for layer_name in layers_to_freeze):
#         param.requires_grad = False
#     else:
#         param.requires_grad = True
newCBF.train(num_epoch=10, num_restart=2, warm_start=False)
newCBF.train(num_epoch=10, num_restart=2, warm_start=False)
newCBF.train(num_epoch=10, num_restart=2, warm_start=False)
newCBF.train(num_epoch=10, num_restart=2, warm_start=False)
newCBF.train(num_epoch=10, num_restart=2, warm_start=False)
newCBF.train(num_epoch=10, num_restart=2, warm_start=False)
# # newCBF.run += 1
# newCBF.train(num_epoch=10, num_restart=8, warm_start=False)
# # newCBF.model.load_state_dict(torch.load('Trained_model/NCBF/NCBF_Obs4.pt'))
#
# # There is a bug in verifier that causes memory error due to too many intersections to verify
# veri_result, num = newCBF.veri.proceed_verification()
