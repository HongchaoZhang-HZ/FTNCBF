import torch
from math import sqrt
from Cases.Case import *

class LinearSat(case):
    '''
    Define Quadcopters model
    f(x, u) = [x(3); x(4); x(5);
            (-x(2)/mass) *(u(0)+u(1));
            (1/mass) *(u(0)+u(1))-gravity;
            (length/inertia) *(u(0)-u(1))]
    '''
    def __init__(self):
        DOMAIN = [[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]]
        CTRLDOM = [[-2, 2], [-2, 2], [-2, 2]]
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
        MU = 3.986e14
        a = 500e3
        n = sqrt(MU / a ** 3)
        A = torch.Tensor([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [3*n**2, 0, 0, 0, 2*n, 0],
                          [0, 0, 0, -2*n, 0, 0],
                          [0, 0, -n**2, 0, 0, 0]])
        fx_dot = (A @ x.unsqueeze(-1)).squeeze()
        return fx_dot

    def g_x(self, x):
        '''
        Control affine model
        g(x) = [0 0 0;
                0 0 0;
                0 0 0;
                1 0 0;
                0 1 0;
                0 0 1;]
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        B = torch.Tensor([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        gx_dot = B
        return gx_dot

    def h_x(self, x):
        '''
        Define safe region C:={x|h_x(x) >= 0}
        The safe region is a pole centered at (0,0,any) with radius 0.2
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] scalar output in R
        '''
        r = torch.sqrt(torch.sum(x[:, :3]**2, dim=1))
        in_range = (r >= 0.25) & (r <= 1.5)

        # Define the function for the in-range and out-of-range cases
        h_in_range = torch.exp(-1 / (r ** 2))  # You can choose any differentiable function here
        h_out_of_range = -torch.exp(-1 / (r ** 2))  # You can choose any differentiable function here

        # Combine the two cases using a conditional statement
        return torch.where(in_range, h_in_range, h_out_of_range)
        # return -hx
