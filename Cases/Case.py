import scipy
import torch
import sympy as sp

class case:
    '''
    Define base class for different case studies
    Input Basic information including
        DOMAIN: State space
        CTRLDOM: Control space
        discrete: If the system is discrete-time (DT) then True
                  otherwise False for continuous-time (CT)
    Background:
        The system is considered as CT/DT nonlinear control-affine system
        with state 'x' whose dimension is determined by len(DOMAIN).
        The dynamical model is defined as follows:
            DT: x_nxt = f(x) + g(x) * u
            CT: x_dot = f(x) + g(x) * u
    '''
    def __init__(self, DOMAIN: list, CTRLDOM: list, discrete=False):
        '''
        :param DOMAIN: [list] State space
        :param CTRLDOM: [list] Control space
        :param discrete: [bool] If the system is discrete-time then True
        '''
        self.CTRLDOM = CTRLDOM
        self.CTRLDIM = len(self.CTRLDOM)
        self.DOMAIN = DOMAIN
        self.DIM = len(self.DOMAIN)
        self.discrete = discrete

    def f_x(self, x):
        '''
        Control affine model f(x)
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        f_x = self.fx * x
        return f_x

    def g_x(self, x):
        '''
        Control affine model g(x)
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        g_x = self.gx * x
        return g_x

    def xdot(self, x, u):
        '''
        The dynamical model is defined as follows:
            DT: x_nxt = f(x) + g(x) * u
            CT: x_dot = f(x) + g(x) * u
        :param x: [np.array/torch.Tensor] input state x in R^n
        :param u: [np.array/torch.Tensor] control input u in R^m
        :return: [np.array/torch.Tensor] output in R^n
        '''
        xdot = self.f_x(x) + self.g_x(x) @ u
        return xdot