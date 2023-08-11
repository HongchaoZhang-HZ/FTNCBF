import torch

from Cases.Case import *

class CExample(case):
    '''
    Define classical control case Obstacle Avoidance
    x0_dot = x1 + u
    x1_dot = -x1 + 5 x2
    '''
    def __init__(self):
        DOMAIN = [[-4, 4], [-4, 4]]
        CTRLDOM = [[-2, 2]]
        discrete = False
        super().__init__(DOMAIN, CTRLDOM, discrete=discrete)

    def f_x(self, x):
        '''
        Control affine model f(x)
        f0 = x1
        f1 = -x1 + 5 x2
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''

        x0_dot = x[:, 0]
        x1_dot = -x[:, 0] + 5*x[:, 1]
        x_dot = torch.vstack([x0_dot, x1_dot])
        return x_dot

    def g_x(self, x):
        '''
        Control affine model g(x)=[1 0]'
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''

        g_x0 = torch.ones(len(x))
        g_x1 = torch.zeros(len(x))
        gx = torch.vstack([g_x0, g_x1])
        return gx

    def h_x(self, x):
        '''
        Define safe region C:={x|h_x(x) >= 0}
        The safe region is a pole centered at (0,0,any) with radius 0.2
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] scalar output in R
        '''
        hx = -(x[:, 0]**2 + x[:, 1] ** 2) + 9
        return hx
