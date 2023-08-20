import torch

from Cases.Case import *

class Darboux(case):
    '''
    Define classical control case Darboux
    '''
    def __init__(self):
        '''
        Define classical control case Darboux.
        The system is 2D open-loop nonlinear CT system
        '''
        DOMAIN = [[-2, 2], [-2, 2]]
        CTRLDOM = [[0, 0]]
        discrete = False
        super().__init__(DOMAIN, CTRLDOM, discrete=discrete)

    def f_x(self, x):
        '''
        Control affine model f(x)
        f0 = x0 + 2*x0*x1
        f1 = -x0 + 2*x0^2 - x1^2
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        x0_dot = x[:, 0] + 2 * x[:, 0] * x[:, 1]
        x1_dot = -x[:, 0] + 2 * x[:, 0] ** 2 - x[:, 1] ** 2
        x_dot = torch.vstack([x0_dot, x1_dot])
        return x_dot

    def g_x(self, x):
        '''
        Control affine model g(x)=[0 0 0]'
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        gx = torch.zeros([self.DIM, 1])
        return gx

    def h_x(self, x):
        '''
        Define safe region C:={x|h_x(x) >= 0}
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] scalar output in R
        '''
        hx = (x[:, 0] + x[:, 1] ** 2)
        return hx
