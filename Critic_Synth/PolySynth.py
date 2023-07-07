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
    def __init__(self, case, arch, act_layer, DOMAIN):
        self.case = case
        self.beta = NNet(arch, act_layer, DOMAIN)
        self.theta = NNet(arch, act_layer, DOMAIN)

    def SymPoly(self, coeff, x, degree=4):
        x0 = x[0]
        x1 = x[1]
        exp = 0
        list_name = sorted(itermonomials([x0, x1], degree), key=monomial_key('grlex', [x1, x0]))
        # list_coeff = coeff.reshape(len(list_name))
        list_coeff = coeff[0]
        for idx in range(len(list_name)):
            exp += list_coeff[idx]*list_name[idx]
        return exp

    def PolySOS(self, NCBF):
        x0, x1 = sp.symbols('x0 x1')
        x = [x0, x1]
        # Translate NN to Poly
        poly_name, poly_coeff = NCBF.topolyCBF()
        polyCBF = self.SymPoly(poly_coeff, x)

        # Perform SOS program
        prob = SOSProblem()

        # Compute derivative
        dbdx = [diff(polyCBF, x0), diff(polyCBF, x1)]
        # Define notations
        rho = sp.symbols('rho')
        vrho = prob.sym_to_var(rho)

        # Define Poly and SOS Poly
        alpha = poly_variable('alpha', [x0,x1], 4)
        theta = poly_variable('theta', [x0,x1], 4)
        eta = poly_variable('eta', [x0,x1], 4)
        Lambda = poly_variable('Lambda', [x0,x1], 4)

        # Define const
        fx = self.case.dyn[0]
        gx = self.case.dyn[1]
        const = alpha.as_expr() * (dbdx[0] * fx[0] + dbdx[1] * fx[1]) + \
                theta.as_expr() * (dbdx[0] * gx[0] + dbdx[1] * gx[1]) + \
                eta.as_expr() * polyCBF + rho * Lambda.as_expr() -1
        # const = 1 + eta * polyCBF + rho * Lambda - 1

        # Minimize rho
        # s.t. alpha db/dx f(x) + sum(theta db/dx g(x)) + eta b + rho Lambda -1 >= 0
        prob.add_sos_constraint(Lambda, [x0, x1])
        prob.add_sos_constraint(alpha, [x0, x1])
        prob.add_constraint(prob.sym_to_var(const) >= 0, [x0, x1])
        vrho = prob.sym_to_var(rho)
        prob.set_objective('min', vrho)
        prob.options["primals"] = False
        res = prob.solve()
        # prob.variables.values()

        return alpha, theta, vrho

    def NeuralSOS(self, NCBF, method='NN'):
        flag = False

        # Initial Guess

        # TODO: SOS search for NN
        if method != 'NN':
            alpha, theta, vrho = self.PolySOS(NCBF)

        # Minimize rho
        # s.t. alpha db/dx f(x) + sum(theta db/dx g(x)) + eta b + rho Lambda -1 >= 0
        # rho here is a relaxation variable
        # Two problem to figure out: 1. when Lambda is updated? 2. what is rho?

        return alpha, theta, vrho

    def SOS_Guidance(self):
        # TODO: Formulate loss function to guide NCBF training
        loss = 0
        # Minimize rho
        # s.t. alpha db/dx f(x) + sum(theta db/dx g(x)) + eta b + rho Lambda -1 >= 0
        # rho here is a relaxation variable
        # Two problem to figure out: 1. when Lambda is updated? 2. what is rho?
        return loss

    def dbdxf(self, x):
        # Todo: incorporate Case
        x1 = x[0]
        x2 = x[1]
        x1_dot = x2 + 2 * x1 * x2
        x2_dot = -x1 + 2 * x1 ** 2 - x2 ** 2
        dbdxf = 0 * x1_dot + 1* x2_dot
        return dbdxf

    def poly(self, x, y, n, coeff):
        counter = 0
        fcn = 0
        for nc in range(n + 1):
            for i in range(nc + 1):
                # coeff[counter]
                print("c[", counter, "]"),
                print(" * ", x, "**", i),
                print(" * ", y, "**", nc - i),
                print(" + "),
                counter += 1

    def PolyVerification(self, NCBF):
        x0, x1 = sp.symbols('x0 x1')
        x = [x0, x1]
        # Translate NN to Poly
        poly_name, poly_coeff = NCBF.topolyCBF()
        polyCBF = self.SymPoly(poly_coeff, x)

        # Perform SOS program
        prob = poly_opt_prob([x0, x1], x0+x1)
        # vx0 = prob.sym_to_var(x0)
        # vx1 = prob.sym_to_var(x1)
        res = prob.solve()

        # from picos import Problem
        #
        # P = Problem()
        # x = P.add_variable('x', 2)
        # P.add_constraint(x > [-2, -2])
        # P.add_constraint(x < [2, 2])
        # P.add_constraint(polyCBF==0)
        # P.set_objective("min", x[0]+x[1])
        # P.solve()

        # Compute derivative
        # dbdx = [diff(polyCBF, x0), diff(polyCBF, x1)]
        # res = minimize(self.dbdxf, x0, args=dbdx, tol=1e-6)
        return res


newCBF = NCBF([10, 10], [True, True], [[-2, 2], [-2, 2]])
# newCBF.train(4)
newCBF.model.load_state_dict(torch.load(sys.path[-1]+'/NCBF.pt'))

# Define Case
x0, x1 = sp.symbols('x0, x1')

# hx = (x[0] + x[1] ** 2)
hx = (x0 + x1**2)
# x0dot = x1 + 2*x0*x1
# x1dot = -x0 + 2*x0**2 - x1**2
fx = [x1 + 2*x0*x1,-x0 + 2*x0**2 - x1**2]
gx = [0, 0]
Darboux = case(fx, gx, hx, newCBF.DOMAIN, [])

testCritic = NeuralCritic(Darboux, [10, 10], [True, True], [[-2, 2], [-2, 2]])
# res = testCritic.PolyVerification(newCBF)

visualize(newCBF, True, 5)

# testCritic.PolySOS(newCBF)


# from picos import Problem
#
# P = Problem()
# x = P.add_variable('x', 2)
# P.add_constraint(x > [-2, -2])
# P.add_constraint(x < [2, 2])
# P.add_constraint(self.SymPoly(poly_coeff, x)==0)
# P.set_objective("min", x[0]+x[1])
# P.solve()