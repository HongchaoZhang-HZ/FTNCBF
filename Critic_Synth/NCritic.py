from NNCBF_Synth_V1.Critic_Synth.Gradient_Approx import *
from NNCBF_Synth_V1.Verifier.Verifier import *

import sympy as sp
import numpy as np
from SumOfSquares import SOSProblem, poly_opt_prob, poly_variable
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials
from sympy import diff
from scipy.optimize import minimize, optimize, linprog

class NeuralCritic(NNet):
    def __init__(self, case):
        self.case = case
        self.DOMAIN = self.case.DOMAIN

    def NNGrad(self, target_NCBF, arch, act_layer, num_epoch):
        self.NCBF = target_NCBF
        gradNN_total = []
        for i in range(self.case.DIM):
            gradNN_dim = GradNN(self.NCBF, arch, act_layer, 0)
            gradNN_dim.train(num_epoch)
            gradNN_total.append(gradNN_dim)
        gradNN = torch.hstack(gradNN_total)
        self.gradNN = gradNN

    def ApprroxGrad(self, x, neuralCBF, epsilon, DIM=2):
        Grad = []
        for i in range(DIM):
            refout = neuralCBF.forward(x)
            perturbe = x
            perturbe[i] += epsilon
            difout = neuralCBF.forward(perturbe)
            grad = (difout - refout) / epsilon
            Grad.append(grad)
        gradient = torch.hstack(Grad)
        return gradient

    def feasiblility_regulator(self, neuralCBF, data, epsilon):
        grad_con_list = []
        for item in data:
            gradients = self.ApprroxGrad(item, neuralCBF, epsilon)
            grad_con = gradients @ torch.vstack(self.case.dyn[0](item))
            grad_con_list.append(grad_con)
        grad_condition = torch.hstack(grad_con_list)
        return grad_condition



# newCBF = NCBF([10, 10], [True, True], [[-2, 2], [-2, 2]])
# # newCBF.train(4)
# newCBF.model.load_state_dict(torch.load(sys.path[-1]+'/NCBF.pt'))
#
# # Define Case
# x0, x1 = sp.symbols('x0, x1')
#
# # hx = (x[0] + x[1] ** 2)
# hx = (x0 + x1**2)
# # x0dot = x1 + 2*x0*x1
# # x1dot = -x0 + 2*x0**2 - x1**2
# fx = [x1 + 2*x0*x1,-x0 + 2*x0**2 - x1**2]
# gx = [0, 0]
# Darboux = case(fx, gx, hx, newCBF.DOMAIN, [])
#
# testCritic = NeuralCritic(Darboux, [10, 10], [True, True], [[-2, 2], [-2, 2]])
# # res = testCritic.PolyVerification(newCBF)
#
# visualize(newCBF, True, 5)
