from Verifier import *

class Stochastic_Verifier(Verifier):
    '''
    Define verifier for stochastic NCBFs
    '''
    def __init__(self, NCBF, EKF, delta_gamma, case, grid_shape, verbose=True):
        super().__init__(NCBF, case, grid_shape, verbose=verbose)
        # SNCBF use EKF estimator
        self.EKF = EKF
        self.EKFGain = EKF.K
        self.delta_gamma = delta_gamma
        self.gamma = 0.1
        self.c = torch.diag(torch.ones(self.DIM))
        self.b_gamma = self.NN - self.gamma * self.delta_gamma

    def dbdxf(self, x, W_overl):
        # stochastic version
        fx = self.case.f_x(torch.Tensor(x).reshape([1, self.DIM])).numpy()
        dbdxf = W_overl @ fx

        EKF_term = W_overl @ self.EKFGain @ self.c
        stochastic_term = self.gamma * EKF_term.norm(dim=1)

        return dbdxf - stochastic_term