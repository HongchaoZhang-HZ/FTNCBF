import torch

from Cases.Case import *

class ObsAvoid(case):
    def __init__(self):
        DOMAIN = [[-2, 2], [-2, 2], [-2, 2]]
        CTRLDOM = [-2, 2]
        discrete = False
        self.v = 1
        super().__init__(DOMAIN, CTRLDOM, discrete=discrete)

    def f_x(self, x):
        # x0_dot = v sin(phi)
        # x1_dot = v cos(phi)
        # phi_dot = 0
        v = self.v
        x0_dot = v * torch.sin(x[:, 2])
        x1_dot = v * torch.cos(x[:, 2])
        phi_dot = torch.zeros([len(x)])
        x_dot = torch.vstack([x0_dot, x1_dot, phi_dot])
        return x_dot

    def g_x(self, x):
        # gx = [0, 0, 1]'
        g_x0 = torch.Tensor([0])
        g_x1 = torch.Tensor([0])
        g_phi = torch.Tensor([1])
        gx = torch.vstack([g_x0, g_x1, g_phi])
        return gx

    def h_x(self, x):
        hx = -(x[:, 0]**2 + x[:, 1] ** 2) + 0.04
        return hx
