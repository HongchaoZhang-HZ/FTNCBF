import scipy
import torch
import sympy as sp

class case:
    def __init__(self, fx, gx, hx, DOMAIN, CTRLDOM):
        self.dyn = (fx, gx)
        self.CTRLDOM = CTRLDOM
        self.DOMAIN = DOMAIN
        self.DIM = len(self.DOMAIN)
        self.safe = hx

    def f_x(self, x):
        f_x = self.fx * x
        return f_x

    def g_x(self, x):
        g_x = self.gx * x
        return g_x
    def xdot(self, x, u):
        xdot = self.f_x(x) + self.g_x(x) * u
        return xdot