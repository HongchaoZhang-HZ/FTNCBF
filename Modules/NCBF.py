from Modules.NNet import *
from torch import optim
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials

class NCBF(NNet):
    def __init__(self, arch, act_layer, DOMAIN):
        '''
        Initialize NCBF with a given architecture and ReLU layers
        :param arch: a vector of # of neurons in each layer
        :param act_layer: a vector of # of ReLU layers
        :param DOMAIN: state space domain
        '''
        super().__init__(arch, act_layer, DOMAIN)

    def generate_input(self, shape=[100,100]):
        state_space = self.DOMAIN
        noise = 1e-2 * torch.rand(shape)
        cell_length = (state_space[0][1] - state_space[0][0]) / shape[0]
        nx = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
        ny = torch.linspace(state_space[1][0] + cell_length / 2, state_space[1][1] - cell_length / 2, shape[1])
        vxo, vyo = torch.meshgrid(nx, ny)
        vx = vxo + noise
        vy = vyo + noise
        data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
        data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))
        return vx, vy, data

    def h_x(self, x):
        hx = (x[0] + x[1] ** 2)
        return hx

    def correctness(self, ref_output, model_output, l_co=1):
        '''
        Correctness loss function
        :param ref_output: h(x) output
        :param model_output: nn output
        :param l_co: lagrangian coefficient
        :return: number of correctness violation
        '''
        norm_model_output = torch.tanh(model_output)
        length = len(-ref_output + norm_model_output)
        # norm_ref_output = torch.tanh(ref_output)
        violations = torch.sigmoid((-ref_output + norm_model_output).reshape([1, length]))
        loss = l_co * torch.sum(violations)
        return loss

    def warm_start(self, ref_output, model_output):
        '''
        MSE loss between ref and model
        :param ref_output: h(x) output
        :param model_output: nn output
        :return: MSE
        '''
        # loss = -torch.sum(torch.tanh(output))
        loss = nn.MSELoss()
        loss_fcn = loss(model_output, ref_output)
        return loss_fcn

    def def_loss(self, *loss):
        '''
        Define loss function by adding all loss
        :param loss: *loss allows multiple inputs
        :return: total loss
        '''
        total_loss = 0
        for l in loss:
            total_loss += l
        return total_loss

    def topolyCBF(self, deg=5):
        shape = [100, 100]
        vx, vy, rdm_input = self.generate_input(shape)
        NN_output = self.forward(rdm_input)
        x_train = rdm_input.numpy()
        y_train = NN_output.detach().numpy()

        poly_reg = PolynomialFeatures(degree=deg)
        X_poly = poly_reg.fit_transform(x_train)
        poly_reg.fit(X_poly, y_train)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y_train)
        # return lin_reg, poly_reg
        names = poly_reg.get_feature_names_out()
        coeff = lin_reg.coef_
        return names, coeff

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


    def train(self, num_epoch):
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        for epoch in range(num_epoch):
            # Generate data
            shape = [100,100]
            vx, vy, rdm_input = self.generate_input(shape)

            running_loss = 0.0
            for i, data in enumerate(rdm_input, 0):

                optimizer.zero_grad()

                model_output = self.forward(rdm_input)
                ref_output = torch.tanh(self.h_x(rdm_input.transpose(0, 1)).reshape([shape[0]*shape[1], 1]))

                warm_start_loss = self.warm_start(ref_output, model_output)
                correctness_loss = self.correctness(ref_output, model_output, 1)
                loss = self.def_loss(warm_start_loss + correctness_loss)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0



