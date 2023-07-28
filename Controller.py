import torch

class NCBFCtrl:
    def __init__(self, DIM, CBF):
        self.state_dim = DIM
        self.CBF = CBF

    @property
    def compute_u(self, state):
        return

    @property
    def CBF_based_u(self, state):
        # compute based on self.CBF
        return
