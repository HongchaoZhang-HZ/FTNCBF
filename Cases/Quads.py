import torch

from Cases.Case import *

class Quads(case):
    '''
    Define Quadcopters model
    f(x, u) = [x(3); x(4); x(5);
            (-x(2)/mass) *(u(0)+u(1));
            (1/mass) *(u(0)+u(1))-gravity;
            (length/inertia) *(u(0)-u(1))]
    '''
    def __init__(self):
        DOMAIN = [[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]]
        CTRLDOM = [[-2, 2], [-2, 2]]
        self.length = 0.25
        self.mass = 0.486
        self.inertia = 0.00383
        self.gravity = 9.81
        discrete = False
        super().__init__(DOMAIN, CTRLDOM, discrete=discrete)

    def f_x(self, x):
        '''
        Control affine model f(x)
        f(x) = [x(3); x(4); x(5);
                0; -gravity; 0]
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''

        x0 = x[:, 3]
        x1 = x[:, 4]
        x2 = x[:, 5]
        x3 = torch.zeros(len(x))
        x4 = -self.gravity * torch.ones(len(x))
        x5 = torch.zeros(len(x))
        x_dot = torch.vstack([x0, x1, x2, x3, x4, x5])
        return x_dot

    def g_x(self, x):
        '''
        Control affine model
        g(x) = [0                   0;
                0                   0;
                0                   0;
                (-x(3)/mass)        (-x(3)/mass);
                (1/mass)            (1/mass);
                (length/inertia)    -(length/inertia)]
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        g_x0 = torch.zeros([2, len(x)])
        g_x1 = torch.zeros([2, len(x)])
        g_x2 = torch.zeros([2, len(x)])
        g_x3 = torch.vstack([-x[:, 3]/self.mass, -x[:, 3]/self.mass])
        g_x4 = torch.vstack([1/self.mass * torch.ones([len(x)]), 1/self.mass * torch.ones([len(x)])])
        g_x5 = torch.vstack([self.length/self.inertia * torch.ones([len(x)]),
                             -self.length/self.inertia * torch.ones([len(x)])])
        gx = torch.dstack([g_x0, g_x1, g_x2, g_x3, g_x4, g_x5])
        return gx.reshape([self.DIM, len(x), self.CTRLDIM])

    def h_x(self, x):
        '''
        Define safe region C:={x|h_x(x) >= 0}
        The safe region is a pole centered at (0,0,any) with radius 0.2
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] scalar output in R
        '''
        hx = (x[:, 2]**2 ) - 1
        return -hx
