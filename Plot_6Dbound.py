import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
import numpy as np
from mayavi import mlab
from mayavi.mlab import *

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# code for plotting cube: from
# https://stackoverflow.com/questions/26888098/draw-cubes-with-colour-intensity-with-python
def mlab_plt_cube(xmin, xmax, ymin, ymax, zmin, zmax, c_color):
    def cube_faces(xmin, xmax, ymin, ymax, zmin, zmax):
        faces = []

        x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
        z = np.ones(y.shape) * zmin
        faces.append((x, y, z))

        x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
        z = np.ones(y.shape) * zmax
        faces.append((x, y, z))

        x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
        y = np.ones(z.shape) * ymin
        faces.append((x, y, z))

        x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
        y = np.ones(z.shape) * ymax
        faces.append((x, y, z))

        y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
        x = np.ones(z.shape) * xmin
        faces.append((x, y, z))

        y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
        x = np.ones(z.shape) * xmax
        faces.append((x, y, z))

        return faces

    def trans_x(origin_x):
        k = (64 - 1) / (2 - (-2))
        c = 1 - k * (-2)
        return k * origin_x + c

    def trans_y(origin_y):
        k = (64 - 1) / (2 - (-2))
        c = 1 - k * (-2)
        return k * origin_y + c

    def trans_z(origin_z):
        k = (64 - 1) / (2 + 2)
        c = 1 - k * (-2)
        return k * origin_z + c

    faces = cube_faces(trans_x(xmin), trans_x(xmax), trans_y(ymin), trans_y(ymax), trans_z(zmin), trans_z(zmax))
    for grid in faces:
        x, y, z = grid
        mlab.mesh(x, y, z, color=c_color, opacity=0.3)


# plot sphere
def mlab_plt_sphere(center_x, center_y, center_z, s_rad, s_color):
    x, y, z = np.ogrid[(-2): (2): complex(0, 64), \
              (-2): (2): complex(0, 64), \
              (-2): (2): complex(0, 64)]

    sphere_scalar = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) + (z - center_z) * (z - center_z)
    sphere = mlab.contour3d(sphere_scalar, contours=[s_rad * s_rad], color=s_color, opacity=0.3)


# plot sphere
def mlab_plt_cylinder(center_x, center_y, s_rad, s_color):
    extent = [-0.2, 0.2, -0.2, 0.2, -2, 2]
    x, y, z = np.ogrid[(-2): (2): complex(0, 64), \
              (-2): (2): complex(0, 64), \
              (-2): (2): complex(0, 64)]

    cylinder_scalar = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) - 0 * z
    cylinder = mlab.contour3d(cylinder_scalar, contours=[s_rad * s_rad], color=s_color, opacity=0.3, extent=extent)


# plot initial

# plot unsafe
# def mlab_plt_unsafe():
#     if len(prob.SUB_UNSAFE) == 0:
#         if prob.UNSAFE_SHAPE == 1:  # cube
#             mlab_plt_cube(prob.UNSAFE[0][0], prob.UNSAFE[0][1], prob.UNSAFE[1][0], prob.UNSAFE[1][1], prob.UNSAFE[2][0],
#                           prob.UNSAFE[2][1], (1, 0, 0))
#         elif prob.UNSAFE_SHAPE == 2:  # sphere
#             mlab_plt_sphere((prob.UNSAFE[0][0] + prob.UNSAFE[0][1]) / 2, (prob.UNSAFE[1][0] + prob.UNSAFE[1][1]) / 2, \
#                             (prob.UNSAFE[2][0] + prob.UNSAFE[2][1]) / 2, (prob.UNSAFE[0][1] - prob.UNSAFE[0][0]) / 2,
#                             (1, 0, 0))
#         elif prob.UNSAFE_SHAPE == 3:  # cylinder
#             mlab_plt_cylinder((prob.UNSAFE[0][0] + prob.UNSAFE[0][1]) / 2, (prob.UNSAFE[1][0] + prob.UNSAFE[1][1]) / 2, \
#                               (prob.UNSAFE[0][1] - prob.UNSAFE[0][0]) / 2, (1, 0, 0))
#         else:
#             x, y, z = np.ogrid[(prob.DOMAIN[0][0]): (prob.DOMAIN[0][1]): complex(0, superp.PLOT_LEN_B[0]), \
#                       (prob.DOMAIN[1][0]): (prob.DOMAIN[1][1]): complex(0, superp.PLOT_LEN_B[1]), \
#                       (prob.DOMAIN[2][0]): (prob.DOMAIN[2][1]): complex(0, superp.PLOT_LEN_B[2])]
#             hyperplane_scalar = x + y + z

            # hyperplane = mlab.contour3d(hyperplane_scalar, contours=[2], color=(1, 0, 0), opacity=0.3, extent=extent)
#
#     else:
#         for i in range(len(prob.SUB_UNSAFE)):
#             curr_shape = prob.SUB_UNSAFE_SHAPE[i]
#             curr_range = prob.SUB_UNSAFE[i]
#             if curr_shape == 1:  # cube
#                 mlab_plt_cube(curr_range[0][0], curr_range[0][1], curr_range[1][0], curr_range[1][1], curr_range[2][0],
#                               curr_range[2][1], (1, 0, 0))
#             elif curr_shape == 2:  # sphere
#                 mlab_plt_sphere((curr_range[0][0] + curr_range[0][1]) / 2, (curr_range[1][0] + curr_range[1][1]) / 2, \
#                                 (curr_range[2][0] + curr_range[2][1]) / 2, (curr_range[0][1] - curr_range[0][0]) / 2,
#                                 (1, 0, 0))
#             else:  # cylinder
#                 mlab_plt_cylinder((curr_range[0][0] + curr_range[0][1]) / 2, (curr_range[1][0] + curr_range[1][1]) / 2, \
#                                   (curr_range[0][1] - curr_range[0][0]) / 2, (1, 0, 0))


# generating plot data for nn
def gen_plot_data():
    sample_x = torch.linspace(-2, 2, int(64))
    sample_y = torch.linspace(-2, 2, int(64))
    sample_z = torch.linspace(-2, 2, int(64))
    grid_xyz = torch.meshgrid([sample_x, sample_y, sample_z])
    flatten_xyz = [torch.flatten(grid_xyz[i]) for i in range(len(grid_xyz))]
    plot_input = torch.stack(flatten_xyz, 1)

    return plot_input


# plot barrier
def mlab_plt_barrier(model):
    # generating nn_output for plotting
    plot_input = gen_plot_data()
    nn_output = model(torch.hstack([plot_input, torch.Tensor([-2, -2,  -2])*torch.ones([262144, 3])]))
    plot_output = (nn_output[:, 0]).reshape(64, 64, 64)
    # barrier_plot = mlab.contour3d(plot_output.detach().numpy(), contours = [-superp.TOL_BOUNDARY, 0, -superp.TOL_BOUNDARY], \
    # 								color = (1, 1, 0), opacity = 0.3) # yellow
    extent = [-0.6, 0.6, -0.6, 0.6, -0.6, 0.6]
    src = mlab.pipeline.scalar_field(plot_output.detach().numpy(), extent=extent)
    barrier_plot = mlab.pipeline.iso_surface(src, contours=[0.02, 0.02, 0.02], \
                                             color=(1, 1, 0), opacity=1, extent=extent)

    axes = mlab.axes(xlabel='$p_x$', ylabel='$p_y$', zlabel='$p_z$')
    axes.label_text_property.font_family = 'times'
    axes.label_text_property.font_size = 30
    axes.axes.font_factor = 2


# plot frame
def mlab_plt_frame():
    x, y, z = np.mgrid[(-2): (2): complex(0, 64), \
              (-2): (2): complex(0, 64), \
              (-2): (2): complex(0, 64)]
    u = 0 * x
    v = 0 * y
    w = 0 * z
    src = mlab.pipeline.vector_field(u, v, w)
    frame = mlab.pipeline.vectors(src, scale_factor=0, opacity=0)
    # frame = mlab.quiver3d(u, v, w, color = (1, 1, 1), scale_factor = 0, opacity = 0.0)
    # print(src.get_output_dataset())
    # input
    # mlab.outline()
    # mlab.axes(xlabel='$p_x$', ylabel='$p_y$', zlabel='$p_z$')
    # mlab.axes.font_factor=2
    axes = mlab.axes(xlabel='$p_x$', ylabel='$p_y$', zlabel='$p_z$')
    axes.label_text_property.font_family = 'times'
    axes.label_text_property.font_size = 30
    axes.axes.font_factor = 2


# plot 3d-system
def plot_barrier_3d(model):
    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab_plt_barrier(model)
    # mlab_plt_init()
    # mlab_plt_unsafe()
    # Create a meshgrid of spherical coordinates
    phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]

    # Convert spherical coordinates to Cartesian coordinates
    x = 0.25 * np.sin(theta) * np.cos(phi)
    y = 0.25 * np.sin(theta) * np.sin(phi)
    z = 0.25 * np.cos(theta)

    # Plot the sphere
    mlab.mesh(x, y, z, color=(1, 0, 0))
    # mlab_plt_sphere((-0.25 + 0.25) / 2, (-0.25 + 0.25) / 2, \
    #                 (-0.25 + 0.25) / 2, (0.5) / 2,
    #                 (1, 0, 0))
    # mlab_plt_vector()
    # mlab_plt_flow()
    # mlab_plt_frame()
    mlab.show()

from Cases.LinearSat import LinearSat
LinearSat = LinearSat()
from SoloSNCBFLinearSat import SNCBF_Synth
gamma_list = [0.001, 0.002, 0.0015, 0.001, 0.01, 0.001, 0.01, 0.001]
newCBF = SNCBF_Synth([128, 128], [True, True], LinearSat,
                     sigma=[0.001, 0.001, 0.001, 0.00001, 0.0001, 0.00001, 0.00001, 0.00001],
                     nu=[0.001, 0.001, 0.001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
                     gamma_list=gamma_list,
                     verbose=True)
newCBF.model.load_state_dict(torch.load('Trained_model/NCBF/linearSat30.pt'))
plot_barrier_3d(newCBF.model)