from NCBF import *
import cma
from cmaes import CMA
from collections import OrderedDict
from Critic_Synth.NCritic import *

class NCBF_Synth(NCBF):
    def __init__(self,arch, act_layer, DOMAIN, case):
        super().__init__(arch, act_layer, DOMAIN)
        self.critic = NeuralCritic(case)
        self.veri = Verifier(NCBF=self, case=case, grid_shape=[100, 100], verbose=False)


    def feasibility_loss(self, model_output, grad_condition, l_co=1):
        # when model_output is close to the boundary grad condition comes in
        violations = grad_condition.reshape(torch.sigmoid(model_output).shape) * torch.sigmoid(model_output)
        # loss = torch.sum(torch.sigmoid(-violations).reshape([1, 10000]))
        violations = torch.max((-violations).reshape([1, 10000]), torch.zeros([1, 10000]))
        loss = l_co * torch.sum(violations)
        return loss

    def safe_correctness(self, ref_output, model_output, l_co=1):
        norm_model_output = torch.tanh(model_output)
        length = len(-ref_output + norm_model_output)
        # norm_ref_output = torch.tanh(ref_output)
        FalsePositive_loss = torch.max(-ref_output.reshape([1, length]), torch.zeros([1, length])) * \
                             torch.max((model_output + 0.01).reshape([1, length]), torch.zeros([1, length]))
        FalseNegative_loss = torch.max(ref_output.reshape([1, length]), torch.zeros([1, length])) * \
                             torch.max((-model_output + 0.01).reshape([1, length]), torch.zeros([1, length]))
        loss = l_co * (FalsePositive_loss + FalseNegative_loss)
        return loss

    def train(self, num_epoch):
        optimizer = optim.SGD(self.model.parameters(), lr=1e-4)

        for epoch in range(num_epoch):
            # Generate data
            shape = [100,100]
            vx, vy, rdm_input = self.generate_input(shape)

            running_loss = 0.0
            for i in range(1000):

                optimizer.zero_grad()

                model_output = self.forward(rdm_input)
                ref_output = torch.tanh(self.h_x(rdm_input.transpose(0, 1)).reshape([shape[0]*shape[1], 1]))

                warm_start_loss = self.warm_start(ref_output, model_output)
                correctness_loss = self.safe_correctness(ref_output, model_output, 1)
                # x[1] + 2 * x[0] * x[1], -x[0] + 2 * x[0] ** 2 - x[1] ** 2
                dx0data = rdm_input + torch.Tensor([0.001, 0])
                dx1data = rdm_input + torch.Tensor([0, 0.001])
                feasibility_output = (self.forward(dx0data)-model_output).reshape([10000]) * \
                                     rdm_input[:,0] + 2*rdm_input[:,0]*rdm_input[:,1] \
                                     + self.forward(dx1data)-model_output * \
                                     (-rdm_input[:,0] + 2*rdm_input[:,0]**2 - rdm_input[:,1]**2)
                check_item = torch.max((-torch.abs(model_output)+0.2).reshape([1, 10000]), torch.zeros([1, 10000]))
                # feasibility_loss = torch.sum(torch.tanh(check_item*feasibility_output))
                violations = -check_item * feasibility_output
                # feasibility_loss = torch.sum(torch.sigmoid(violations))
                feasibility_loss = torch.max((-feasibility_output - model_output), torch.zeros([1, 10000]))
                # feasibility_output = self.critic.feasiblility_regulator(self.model, rdm_input, epsilon=0.01)
                # feasibility_loss = self.feasibility_loss(model_output, feasibility_output, 1)
                loss = self.def_loss(correctness_loss + feasibility_loss)
                # loss = self.def_loss(warm_start_loss + correctness_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # if i % 1000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            visualize(self.model)

    def batch_train(self, num_epoch):
        # optimizer = optim.SGD(self.model.parameters(), lr=1e-1)
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(num_epoch):
            penalty = 1
            shape = [100, 100]
            # Generate data
            vx, vy, rdm_input = self.generate_input(shape)
            for episode in range(100):
                running_loss = 0.0
                for i, data in enumerate(rdm_input, 0):

                    optimizer.zero_grad()

                    model_output = self.forward(data)
                    ref_output = torch.tanh(self.h_x(data))

                    warm_start_loss = self.warm_start(ref_output, model_output)
                    correctness_loss = self.safe_correctness(ref_output, model_output, 1)
                    # x[1] + 2 * x[0] * x[1], -x[0] + 2 * x[0] ** 2 - x[1] ** 2
                    dx0data = data + torch.Tensor([0.0001, 0])
                    dx1data = data + torch.Tensor([0, 0.01])
                    # feasibility_output = (self.forward(dx0data)-model_output).reshape([1]) * \
                    #                      data[0] + 2*data[0]*data[1] \
                    #                      + self.forward(dx1data)-model_output * \
                    #                      (-data[0] + 2*data[0]**2 - data[1]**2)
                    feasibility_output = (self.forward(dx0data) - model_output).reshape([1]) * \
                                         data[0] + 2 * data[0] * data[1] \
                                         + self.forward(dx1data) - model_output * \
                                         (-data[0] + 2 * data[0] ** 2 - data[1] ** 2)
                    check_item = torch.max((-torch.abs(model_output)+0.2).reshape([1, 1]), torch.zeros([1, 1]))
                    violations = -check_item*feasibility_output
                    feasibility_loss = torch.max((-feasibility_output - model_output), torch.zeros([1, 1]))
                    # feasibility_loss = torch.sum(torch.max(violations, torch.zeros(1)))
                    # feasibility_output = self.critic.feasiblility_regulator(self.model, rdm_input, epsilon=0.01)
                    # feasibility_loss = self.feasibility_loss(model_output, feasibility_output, 1)
                    # loss = self.def_loss(warm_start_loss + correctness_loss + penalty * feasibility_loss)
#                      (self.forward(torch.Tensor([-2,0])).item()+1)**2,
                    loss = self.def_loss(correctness_loss + feasibility_loss)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    # if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, episode + 1, running_loss / 10000))
                running_loss = 0.0
                # if episode % 10 == 9:
                #     # visualize(self.model)
                #     veri_result, num = self.veri.proceed_verification()
                #     if correctness_loss==0 and veri_result:
                #         # visualize(self.model)
                #         return
                # else:
                #     penalty = penalty + 1
                veri_result, num = self.veri.proceed_verification()
                # if correctness_loss == 0 and veri_result:
                    # visualize(self.model)
                    # return
                visualize(self.model)

    def load_para(self, paras):
        new_state_dict = OrderedDict()
        # model = NCBF([10, 10], [True, True], [[-2, 2], [-2, 2]])
        # for key in old_state_dict.items():
        new_state_dict['0.weight'] = torch.Tensor(np.asarray(paras[0:20]).reshape([10, 2]))
        new_state_dict['0.bias'] = torch.Tensor(np.asarray(paras[20:30]).reshape([10]))
        new_state_dict['2.weight'] = torch.Tensor(np.asarray(paras[30:130]).reshape([10, 10]))
        new_state_dict['2.bias'] = torch.Tensor(np.asarray(paras[130:140]).reshape([10]))
        new_state_dict['4.weight'] = torch.Tensor(np.asarray(paras[140:150]).reshape([1, 10]))
        new_state_dict['4.bias'] = torch.Tensor(np.asarray(paras[150:151]).reshape([1]))
        self.model.load_state_dict(new_state_dict, strict=True)

    def init_para(self, old_model):
        paras = np.hstack(old_model.state_dict()['0.weight'].numpy())
        paras = np.hstack([paras, old_model.state_dict()['0.bias'].numpy()])
        paras = np.hstack([paras, old_model.state_dict()['2.weight'].numpy().flatten()])
        paras = np.hstack([paras, old_model.state_dict()['2.bias'].numpy()])
        paras = np.hstack([paras, old_model.state_dict()['4.weight'].numpy().flatten()])
        paras = np.hstack([paras, old_model.state_dict()['4.bias'].numpy()])

        return paras

    def CMA_train(self, num_epoch):
        # initial solution
        # x0 = np.zeros(152)
        x0 = self.init_para(self.model)
        optimizer = CMA(mean=x0, sigma=100)
        for epoch in range(num_epoch):
            # Generate data
            vx, vy, rdm_input = self.generate_input([100,100])
            running_loss = 0.0

            for generation in range(100):
                solutions = []
                # import ipdb; ipdb.set_trace()
                for _ in range(optimizer.population_size):
                    x = optimizer.ask()
                    self.load_para(x)

                    model_output = self.forward(rdm_input)
                    ref_output = torch.tanh(self.h_x(rdm_input.transpose(0, 1)).reshape([10000, 1]))

                    warm_start_loss = self.warm_start(ref_output, model_output)
                    correctness_loss = self.correctness(ref_output, model_output, 1)
                    # x[1] + 2 * x[0] * x[1], -x[0] + 2 * x[0] ** 2 - x[1] ** 2
                    dx0data = rdm_input + torch.Tensor([0.001, 0])
                    dx1data = rdm_input + torch.Tensor([0, 0.001])
                    feasibility_output = (self.forward(dx0data) - model_output).reshape([10000]) * \
                                         rdm_input[:, 0] + 2 * rdm_input[:, 0] * rdm_input[:, 1] \
                                         + self.forward(dx1data) - model_output * \
                                         (-rdm_input[:, 0] + 2 * rdm_input[:, 0] ** 2 - rdm_input[:, 1] ** 2)
                    check_item = torch.max((-torch.abs(model_output) + 0.2).reshape([1, 10000]),
                                           torch.zeros([1, 10000]))
                    violations = -check_item * feasibility_output
                    feasibility_loss = torch.sum(torch.sigmoid(violations * feasibility_output))
                    # feasibility_output = self.critic.feasiblility_regulator(self.model, rdm_input, epsilon=0.01)
                    # feasibility_loss = self.feasibility_loss(model_output, feasibility_output, 1)
                    loss = self.def_loss(warm_start_loss + correctness_loss + feasibility_loss).item()/10000
                    # if loss < 0:
                    #     print(correctness_loss)
                    #     print(feasibility_loss)
                    solutions.append((x, loss))
                    print(loss)
                    if loss <= 5001:
                        veri = Verifier(NCBF=self, case=Darboux, grid_shape=[100, 100], verbose=True)
                        verify_res, num = veri.proceed_verification()
                        if verify_res and num > 0:
                            return

                optimizer.tell(solutions)

# Define Case
# x0, x1 = sp.symbols('x0, x1')

hx = lambda x: (x[0] + x[1] ** 2)
# hx = (x0 + x1**2)
# x0dot = x1 + 2*x0*x1
# x1dot = -x0 + 2*x0**2 - x1**2
fx = lambda x: [x[1] + 2*x[0]*x[1],-x[0] + 2*x[0]**2 - x[1]**2]
gx = [0, 0]
Darboux = case(fx, gx, hx, [[-2,2],[-2,2]], [])
newCBF = NCBF_Synth([10, 10], [True, True], [[-2, 2], [-2, 2]], Darboux)
# newCBF.model.load_state_dict(torch.load('NCBF.pt'))
newCBF.batch_train(3)
# res = testCritic.PolyVerification(newCB

visualize(newCBF)