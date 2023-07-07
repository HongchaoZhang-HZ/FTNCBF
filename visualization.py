import matplotlib.pyplot as plt
import numpy as np
# from veri_util import *
import torch
import sympy as sp

def h_x(x):
    hx = (x[0]+x[1]**2)
    return torch.tanh(hx)

def visualize(nnmodel, polyflag=False, polydeg=5):
    state_space = [[-2,2],[-2,2]]
    shape = [100,100]
    cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]
    nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
    ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
    vx, vy = torch.meshgrid(nx, ny)
    data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
    data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))

    if not polyflag:
        output = (nnmodel.forward(data)).detach().numpy()
    else:
        poly_name, poly_coeff = nnmodel.topolyCBF(polydeg)
        x0, x1 = sp.symbols('x0, x1')
        x = [x0, x1]
        fcn = nnmodel.SymPoly(poly_coeff, x, polydeg)
        data = data.detach().numpy()
        output = np.array([fcn.subs({x0: data[_][0], x1: data[_][1]}) for _ in range(len(data))], dtype=np.float)
    # output = h_x(data.transpose(0,1))
    z = output.reshape(shape)
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(vx, vy, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    ax.axis([vx.min(), vx.max(), vy.min(), vy.max()])
    fig.colorbar(c, ax=ax)
    for i in range(100):
        for j in range(100):
            if np.linalg.norm(z[i][j])<0.003:
                plt.scatter(nx[i], ny[j])
    plt.show()

def visualization(nnmodel):
    state_space = [[-2,2],[-2,2]]
    shape = [100,100]
    cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]
    nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
    ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
    vx, vy = torch.meshgrid(nx, ny)
    data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
    data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))

    output = (nnmodel.forward(data)).detach().numpy()
    # output = h_x(data.transpose(0,1))
    z = output.reshape(shape)
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(vx, vy, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    ax.axis([vx.min(), vx.max(), vy.min(), vy.max()])
    fig.colorbar(c, ax=ax)
    for i in range(100):
        for j in range(100):
            if np.linalg.norm(z[i][j])<0.003:
                plt.scatter(nx[i], ny[j])
    plt.show()


def visualization_fcn(NCBF, Critic):
    state_space = [[-2,2],[-2,2]]
    shape = [100,100]
    cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]
    nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
    ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
    vx, vy = torch.meshgrid(nx, ny)
    data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
    data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))

    poly_name, poly_coeff = NCBF.topolyCBF()
    x0, x1 = sp.symbols('x0, x1')
    x = [x0, x1]
    fcn = Critic.SymPoly(poly_coeff, x)
    # output = h_x(data.transpose(0,1))
    data = data.detach().numpy()
    output = np.array([fcn.subs({x0:data[_][0], x1:data[_][1]}) for _ in range(len(data))], dtype=np.float)
    z = output.reshape(shape)
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(vx, vy, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    ax.axis([vx.min(), vx.max(), vy.min(), vy.max()])
    fig.colorbar(c, ax=ax)
    for i in range(100):
        for j in range(100):
            if np.linalg.norm(z[i][j])<0.003:
                plt.scatter(nx[i], ny[j])
    plt.show()