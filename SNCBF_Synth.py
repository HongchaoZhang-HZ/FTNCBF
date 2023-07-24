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
from Verifier import Verifier
from Critic_Synth.NCritic import *
import time
# from collections import OrderedDict

class SNCBF_Synth(NCBF_Synth):
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
        super().__init__(arch, act_layer, case, verbose=False)
        # Under construction: Critic is designed to tuning loss fcn automatically
        # self.critic = NeuralCritic(case)
        # Verifier proposed to verify feasibility
        self.veri = Verifier(NCBF=self, case=case, grid_shape=[100, 100, 100], verbose=verbose)
        lctime = time.ctime(time.time())
        # Tensorboard
        self.writer = SummaryWriter(f'./runs/SNCBF/{lctime}'.format(lctime))
        # todo: in FT-SNCBF these variables need to be chosen based on fault type
        self.gamma = 0.1
        self.c = 1

        self.run = 0

    def numerical_b_gamma(self, grad, gamma):
        # todo: debug
        return np.max(np.abs(grad)) * gamma

    def EKF(self):
        # todo: extended kalman filter gain for different sensor failure
        K = torch.ones([self.DIM, self.DIM])
        return K

    def feasibility_loss(self, grad_vector, X_batch):
        # compute loss based on (db/dx)*fx + (db/dx)*gx*u
        dbdxfx = (grad_vector.transpose(0, 1).unsqueeze(1)
                  @ self.case.f_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        dbdxgx = (grad_vector.transpose(0, 1).unsqueeze(1)
                  @ self.case.g_x(X_batch).transpose(0, 1).unsqueeze(2)).squeeze()
        u = self.feasible_u(dbdxfx, dbdxgx)
        # update delta_gamma
        self.delta_gamma = self.numerical_b_gamma(grad_vector, self.gamma)
        stochastic_term = -self.gamma * torch.linalg.norm(grad_vector @ self.EKF() * self.c)
        feasibility_output = dbdxfx + dbdxgx * u + stochastic_term
        return feasibility_output

    def feasible_violations(self, model_output, feasibility_output, batch_length, rlambda):
        b_gamma = (model_output - self.delta_gamma)
        violations = -1 * feasibility_output - rlambda * torch.abs(b_gamma.transpose(0, 1))
        return violations

ObsAvoid = ObsAvoid()
newCBF = SNCBF_Synth([32, 32], [True, True], ObsAvoid, verbose=True)
# newCBF.train(50, warm_start=True)
# newCBF.run += 1
newCBF.train(num_epoch=10, num_restart=8, warm_start=False)
# newCBF.model.load_state_dict(torch.load('Trained_model/NCBF/NCBF_Obs4.pt'))

# There is a bug in verifier that causes memory error due to too many intersections to verify
veri_result, num = newCBF.veri.proceed_verification()
