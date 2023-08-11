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
from Cases.CExample import CExample
from Verifier.Verifier import Verifier
from Visualization.visualization import visualize, visualization
# from Critic_Synth.NCritic import *
import time
# from collections import OrderedDict
from NCBF_Synth import NCBF_Synth
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from matplotlib import pyplot as plt

def compute_u(x, dt, case):
    # def fcn(u, cx):
    #     xn = cx + dt * (case.f_x(cx) + (case.g_x(cx) @ u).unsqueeze(1)).transpose(0, 1)
    #     return (xn[0][1]-4) ** 2 + xn[0][0]**2 + u**2
    #
    # # minimize ||u||
    # u0 = np.array([0])
    # res = minimize(fcn, u0, args=x)
    # norm_u = -1
    # xe = x - torch.tensor([[3, 1]])
    K = torch.tensor([-12.4252, +64.2673], dtype=torch.float)
    # norm_u = res.x
    norm_u = (K @ torch.tensor(x, dtype=torch.float).transpose(0,1)).item()
    return norm_u

def NCBF_condition(NCBF, case, x):
    dbdx = NCBF.get_grad(x)
    # stochastic version
    fx = case.f_x(torch.Tensor(x).reshape([1, 2])).numpy()
    dbdxf = dbdx @ fx
    return dbdxf

def NCBFcontroller(NCBF, case, x, dt):
    def fcn(u, norm_u):
        return (u-norm_u) ** 2
    # def fcn(u, x):
    #     xn = x + dt * (case.f_x(x) + (case.g_x(x) @ u).unsqueeze(1)).transpose(0, 1)
    #     return (xn[0][0]) ** 2 + (xn[0][1]-4) ** 2
    NCBF_con = NCBF_condition(NCBF, case, x)
    affine = NCBF.get_grad(x)[0] @ case.g_x(x)
    bx = NCBF.model.forward(torch.tensor(x, dtype=torch.float)).item()
    NCBFfcn = lambda u: (affine @ u).squeeze() + (NCBF_con).squeeze() + bx
    NCBFCON = NonlinearConstraint(NCBFfcn, 0, np.inf)
    # minimize ||u||
    norm_u = compute_u(x, dt, case)
    # norm_u = 0
    u0 = np.array([0])
    res = minimize(fcn, u0, args=norm_u, constraints=NCBFCON)
    # print(res.x-norm_u)
    return res.x
    # return norm_u

def saferun(NCBF, case, T, dt, x0):
    # Define initial state x
    # x0 = np.array([[0.0, 0.0]])
    traj = []
    u_tr = []
    x = x0
    traj.append(x)
    # TODO: put these in self.CR.Resolution(res.success)
    for t in range(T):
        # NCBF control optimizer output
        res = NCBFcontroller(NCBF, case, x, dt)
        # res = polycontroller(case, x, dt)
        u = res
        # u = compute_u(x, dt, case)
        # u = np.clip(u, -100, 100)
        u_tr.append(u)
        # if the result has a false feasibility flag,
        # then go step 2

        # Post controller steps

        # update state x in a discrete time manner
        x = torch.tensor(x)
        x = x + dt * (case.f_x(x) + (case.g_x(x) * u)).transpose(0, 1)
        traj.append(x)
    return traj, u_tr


def vis(nnmodel, name='Verified NCBF'):
    state_space = [[-4,4],[-4,4]]
    shape = [100,100]
    cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]
    nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
    ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
    vx, vy = torch.meshgrid(nx, ny)
    data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
    data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))
    output = (nnmodel.forward(data)).detach().numpy()
    z = output.reshape(shape)
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots()

    # c = ax.pcolormesh(vx, vy, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title(name)
    # set the limits of the plot to the limits of the data
    # ax.axis([vx.min(), vx.max(), vy.min(), vy.max()])
    # fig.colorbar(c, ax=ax)
    for i in range(100):
        for j in range(100):
            if np.linalg.norm(z[i][j])<=0.003:
                plt.scatter(nx[i], ny[j], color='blue', alpha=0.8)
    fig.show()


CE = CExample()
x0 = torch.Tensor([[0, 0.1]])
newCBF = NCBF_Synth([32, 32], [True, True], CE, verbose=True)
newCBF.model.load_state_dict(torch.load('NCBF_CE0.pt'))
# newCBF.veri.proceed_verification()
trajN, uN = saferun(newCBF, CE, 22000, 0.0001, x0)
handCBF = NCBF_Synth([4], [True], CE, verbose=True)
from collections import OrderedDict
new_state_dict = OrderedDict()
new_state_dict['0.weight'] = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]])
new_state_dict['0.bias'] = torch.tensor([0, 0, 0, 0])
new_state_dict['2.weight'] = torch.tensor([[-1, -1, -1, -1]])
new_state_dict['2.bias'] = torch.tensor([1])
handCBF.model.load_state_dict(new_state_dict)
# vis(handCBF)
trajh, uh = saferun(handCBF, CE, 22000, 0.0001, x0)
#
state_space = [[-4,4],[-4,4]]
shape = [100,100]
cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]
nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
vx, vy = torch.meshgrid(nx, ny)
data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))
output = (newCBF.forward(data)).detach().numpy()
z = output.reshape(shape)
z_min, z_max = -np.abs(z).max(), np.abs(z).max()
# plt.figure(figsize=(15, 4))
fig, ax = plt.subplots()
ax.set_title('Comparison of trajectory of $b_{\\theta}$ and $b_{c}$.', fontsize=20)
fig.set_size_inches(8,4)
for i in range(100):
    for j in range(100):
        if np.linalg.norm(z[i][j])<=0.003:
            ax.scatter(nx[i], ny[j], color='green')
ax.plot(nx[i], ny[j], color='green', label='Safety Boundary of $b_\\theta$', linewidth=4)
output = (handCBF.forward(data)).detach().numpy()
z = output.reshape(shape)
z_min, z_max = -np.abs(z).max(), np.abs(z).max()
for i in range(100):
    for j in range(100):
        if np.linalg.norm(z[i][j])<=0.1:
            ax.scatter(nx[i], ny[j], color='grey')
ax.plot(nx[i], ny[j], color='grey', label='Safety Boundary of $b_c$', linewidth=4)
ax.plot(np.array(trajN).squeeze(1)[:, 0], np.array(trajN).squeeze(1)[:, 1], label='NCBF Trajectory', linewidth=4)
ax.plot(np.array(trajh).squeeze(1)[:, 0], np.array(trajh).squeeze(1)[:, 1], label='CE Trajectory', linewidth=4)
theta = np.linspace(0, 2 * np.pi, 150)
radius = 3.2
a = radius * np.cos(theta)
b = radius * np.sin(theta)
# plt.Circle((0, 0), 3, fill=False, color='red')
ax.plot(a, b, color='red', linewidth=5)
ax.plot(3, 0, color='red', label='Safety Boundary of $h(x)$', linewidth=4)
plt.xlim([-4, 4])
plt.ylim([-4, 4])
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.5, pos.height])
ax.legend(loc='center right', bbox_to_anchor=(1.68, 0.5), fontsize='18')
plt.show()
# plt.plot(np.array(trajN)[:,0,1],label='Verified NCBF')
# plt.plot(np.array(trajh)[:,0,1],label='Counter Example')
# plt.scatter(21,np.array(trajh)[21,0,1],color='r')
# plt.text(7, 3.2, 'enter unsafe region', fontsize=14, bbox=dict(facecolor='red', alpha=0.5))

# # vis(newCBF)
# vis(handCBF)

# stime = time.time()
# for i in range(10):
#     trajN, uN = saferun(newCBF, CE, 30, 0.01, torch.tensor([[0.0, 1.0]]))
# etime = time.time()







# def poly_condition(case, x):
#     dbdx = 2*x.norm()*x - x.norm()
#     # stochastic version
#     fx = case.f_x(torch.Tensor(x).reshape([1, 2])).numpy()
#     dbdxf = dbdx @ fx
#     return dbdxf
#
# def polycontroller(case, x, dt):
#     def fcn(u, norm_u):
#         return (u-norm_u) ** 2
#     # def fcn(u, x):
#     #     xn = x + dt * (case.f_x(x) + (case.g_x(x) @ u).unsqueeze(1)).transpose(0, 1)
#     #     return (xn[0][0]) ** 2 + (xn[0][1]-4) ** 2
#     poly_con = poly_condition(case, x)
#     affine = (2*x.norm()*torch.tensor(x,dtype=torch.float)- x.norm()) @ case.g_x(x)
#     polyfcn = lambda u: (affine @ u).squeeze() + (poly_con).squeeze()
#     polyCON = NonlinearConstraint(polyfcn, 0, np.inf)
#     # minimize ||u||
#     norm_u = compute_u(x)
#     u0 = np.array([0])
#     res = minimize(fcn, u0, args=norm_u, constraints=polyCON)
#     return res