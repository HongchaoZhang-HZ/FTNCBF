import sys
sys.path.append('..')
from NNCBF_Synth_V1.NCBF import *

# Brutal force approximation of numerical gradient
def ApprroxGrad(x, neuralCBF, epsilon, DIM=2):
    Grad = []
    for i in range(DIM):
        refout = neuralCBF.forward(x)
        perturbe = x
        perturbe[i] += epsilon
        difout = neuralCBF.forward(perturbe)
        grad = (difout-refout)/epsilon
        Grad.append(grad)
    gradient = np.asarray(Grad)
    return gradient

# NN approximate gradient of NCBF
class GradNN(NCBF):
    def __init__(self, target_NCBF, arch, act_layer, axis):
        self.DOMAIN = target_NCBF.DOMAIN
        self.target = target_NCBF
        self.DIM = len(self.DOMAIN)
        self.act_fun = nn.ReLU()
        # Todo: Modify this part to incorporate multi-dim output
        self.NN = NNet(arch, act_layer, [[-2, 2], [-2, 2]])
        self.model = self.NN.model
        self.axis = axis

    # Generate x_train y_train
    def generate_y_train(self, vx, vy, shape):
        perturbe = 1e-2 * torch.rand(shape)
        data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
        data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))
        ref_output0 = self.target.forward(data)

        if self.axis == 0:
            vx1 = vx + perturbe
            vy1 = vy
        else:
            vx1 = vx
            vy1 = vy + perturbe
        data1 = np.dstack([vx1.reshape([shape[0], shape[1], 1]), vy1.reshape([shape[0], shape[1], 1])])
        data1 = torch.Tensor(data1.reshape(shape[0] * shape[1], 2))
        ref_output1 = self.target.forward(data1)

        grad = (ref_output1 - ref_output0) / perturbe.reshape([shape[0]*shape[1],1])
        return grad

    # train NN
    def train(self, num_epoch):
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001)

        for epoch in range(num_epoch):
            # Generate data
            shape = [100, 100]
            vx, vy, rdm_input = self.generate_input(shape)

            running_loss = 0.0
            for i, data in enumerate(rdm_input, 0):
                optimizer.zero_grad()

                model_output = self.forward(rdm_input)
                grad = self.generate_y_train(vx, vy, shape)
                warm_start_loss = self.warm_start(grad, model_output)
                loss = self.def_loss(warm_start_loss)

                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0



# newCBF = NCBF([10, 10], [True, True], [[-2, 2], [-2, 2]])
# newCBF.train(1)
# CBFx0grad = GradNN(newCBF, [30, 30], [True, True], 0)
# CBFx0grad.train(4)
# CBFx1grad = GradNN(newCBF, [30, 30], [True, True], 1)
# CBFx1grad.train(4)
# visualization(CBFx0grad)
# visualization(CBFx1grad)