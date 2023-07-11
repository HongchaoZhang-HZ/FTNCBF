import torch

from Cases.Case import *

class Darboux(case):
    def __init__(self):
        DOMAIN = [[-2, 2], [-2, 2]]
        CTRLDOM = []
        discrete = False
        super().__init__(DOMAIN, CTRLDOM, discrete=discrete)

    def f_x(self, x):
        x0_dot = x[:, 0] + 2 * x[:, 0] * x[:, 1]
        x1_dot = -x[:, 0] + 2 * x[:, 0] ** 2 - x[:, 1] ** 2
        x_dot = torch.vstack([x0_dot, x1_dot])
        return x_dot

    def g_x(self, x):
        gx = x @ torch.zeros([self.DIM])
        return gx

    def h_x(self, x):
        hx = (x[:, 0] + x[:, 1] ** 2)
        return hx
