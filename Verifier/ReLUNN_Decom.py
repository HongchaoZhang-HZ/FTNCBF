import sys
sys.path.append('..')
from Modules.NCBF import *
import torch.nn.functional as F
from Visualization.visualization import *
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize, optimize, linprog
from scipy.optimize import LinearConstraint, NonlinearConstraint

class ReLUNN_Decom():
    def __init__(self, NCBF, grid_shape, verbose=True):
        self.NN = NCBF
        self.model = self.NN.model
        self.DOMAIN = NCBF.DOMAIN
        self.DIM = NCBF.DIM
        self.shape = grid_shape
        self.cell_length = (self.DOMAIN[0][1] - self.DOMAIN[0][0]) / self.shape[0]
        self.layers = self.get_layers()
        self.act_layer = self.NN.act_layer
        self.num_act_layer = len(self.act_layer)
        self.verbose = verbose

    def gridify(self):
        nx = torch.linspace(self.DOMAIN[0][0] + self.cell_length / 2,
                            self.DOMAIN[0][1] - self.cell_length / 2, self.shape[0])
        ny = torch.linspace(self.DOMAIN[1][0] + self.cell_length / 2,
                            self.DOMAIN[1][1] - self.cell_length / 2, self.shape[1])
        vx, vy = torch.meshgrid(nx, ny)
        data = np.dstack([vx.reshape([self.shape[0], self.shape[1], 1]),
                          vy.reshape([self.shape[0], self.shape[1], 1])])
        data = torch.Tensor(data.reshape(self.shape[0] * self.shape[1], 2))
        return data

    def sect_search(self, data):
        sec_model = BoundedModule(self.model, data)
        ptb = PerturbationLpNorm(norm=np.inf, eps=self.cell_length / 2)
        my_input = BoundedTensor(data, ptb)
        lb, ub = sec_model.compute_bounds(x=(my_input,), method="backward")
        return lb, ub

    def weighted_bound(self, weight, bias, prev_upper, prev_lower):
        prev_mu = (prev_upper + prev_lower) / 2
        prev_r = (prev_upper - prev_lower) / 2
        # Data type change to Double
        dprev_mu = prev_mu.type(torch.double)
        dprev_r = prev_r.type(torch.double)
        dweight = weight.type(torch.double)
        dbias = bias.type(torch.double)
        # IBP
        mu = F.linear(dprev_mu, dweight, dbias)
        r = F.linear(dprev_r, torch.abs(dweight))
        upper = mu + r
        lower = mu - r
        return upper, lower

    def list_flip(self, upper, lower):
        ind = (torch.sign(lower) * torch.sign(upper) - 1) / 2
        idx = torch.nonzero(ind)
        return ind, idx

    def get_layers(self):
        Layers = []
        children = list(self.model.eval())
        count_children = np.size(children)
        for i in range(count_children):
            layer = torch.nn.Sequential(*list(self.model.eval())[:i])
            Layers.append(layer)
        return Layers

    def output_forward_activation(self, input, layer_w, layer_a):
        out_w = layer_w(input)
        out_a = layer_a(input)
        # Find activated neurons in each layer
        activated = torch.eq(out_w, out_a)
        # Reshape the activated vector to [len(activated), 1] for latter multiply
        activated = torch.reshape(activated, [len(activated), 1])
        return out_w, out_a, activated

    def activated_weight_bias(self, activated_set):
        W_list = []
        r_list = []
        para_list = list(self.model.state_dict())
        i = 0
        while i < (len(para_list)):
            weight = self.model.state_dict()[para_list[i]]
            i += 1
            bias = self.model.state_dict()[para_list[i]]
            i += 1
            W_list.append(weight)
            r_list.append(bias)
        # compute the activated weight of the layer
        W_l = torch.mul(activated_set, W_list[0])
        W_overl = torch.matmul(W_list[1], W_l)  # compute \overline{W}(S)
        # compute the activated bias of the layer
        r_l = torch.mul(activated_set, torch.reshape(r_list[0], [len(r_list[0]), 1]))
        r_overl = torch.matmul(W_list[1], r_l) + r_list[1]  # compute \overline{r}(S)
        # compute region/boundary weight
        W_a = W_l
        r_a = r_l
        B_act = [W_a, r_a]  # W_a x <= r_a
        W_i = W_list[0] - W_l

        r_i = -torch.reshape(r_list[0], [len(r_list[0]), 1]) + r_l
        B_inact = [W_i, r_i]  # W_a x <= r_a
        return W_overl, r_overl, B_act, B_inact

    def list_activated(self):
        # Grid the space
        data = self.gridify()

        # Call auto_LiRPA and return cells across.
        lb, ub = self.sect_search(data)

        # Print number of cells
        sect_ind = (torch.sign(lb) * torch.sign(ub) - 1) / 2
        sect_idx = torch.nonzero(sect_ind.reshape([sect_ind.size()[0]]))
        sections = data[sect_idx]
        num_sec = torch.sum(sect_ind)
        if self.verbose:
            print(f'There are', abs(num_sec.item()), 'cells to be checked')

        # 3. For grids that use IBP (use, find '?')
        activated_sets = []
        for cell in data[sect_idx]:
            bounds = cell.transpose(0, 1) + self.cell_length / 2 * \
                     np.array([[-1, 1] for _ in range(len(self.DOMAIN))])
            prev_lower = bounds[:, 0]
            prev_upper = bounds[:, 1]
            weight = self.model.state_dict()['0.weight']
            bias = self.model.state_dict()['0.bias']

            for num_layer in range(self.num_act_layer):
                upper, lower = self.weighted_bound(weight, bias, prev_upper, prev_lower)
                act_ind, act_set = self.list_flip(upper, lower)
                prev_lower = upper
                prev_upper = lower
                weight = self.model.state_dict()['{}.weight'.format(2 * (num_layer + 1))]
                bias = self.model.state_dict()['{}.bias'.format(2 * (num_layer + 1))]

                if num_layer == 0:
                    act_set_layers = act_set
                    num = torch.sum(torch.abs(act_ind))
                else:
                    temp_loc = np.sum(self.NN.arch[:num_layer])
                    act_set_layers = np.vstack([act_set_layers, (act_set + temp_loc)])
                    if num_layer != self.num_act_layer:
                        num += torch.sum(torch.abs(act_ind))
                # num = torch.sum(torch.abs(act_ind))
            activated_sets.append(act_set_layers)

        return activated_sets, sections

    def activated_weight_bias_ml(self, activated_set, num_neuron):
        # Todo: incorporate NN.arch to replace num_neuron
        W_list = []
        r_list = []
        para_list = list(self.model.state_dict())
        i = 0
        while i < (len(para_list)):
            weight = self.model.state_dict()[para_list[i]]
            i += 1
            bias = self.model.state_dict()[para_list[i]]
            i += 1
            W_list.append(weight)
            r_list.append(bias)
        # compute the activated weight of the layer
        for l in range(len(self.act_layer)):
            # compute region/boundary weight

            if l == 0:
                W_l = torch.mul(activated_set[num_neuron * l:num_neuron * (l + 1)], W_list[l])
                r_l = torch.mul(activated_set[num_neuron * l:num_neuron * (l + 1)],
                                torch.reshape(r_list[l], [len(r_list[l]), 1]))
                W_a = W_l
                r_a = r_l
                W_i = W_list[l] - W_l
                r_i = -torch.reshape(r_list[l], [len(r_list[l]), 1]) + r_l
            else:
                W_pre = W_list[l] @ W_l
                r_pre = W_list[l] @ r_l + r_list[l].reshape([len(r_list[l]), 1])
                W_l = activated_set[num_neuron * l:num_neuron * (l + 1)] * W_pre
                r_l = activated_set[num_neuron * l:num_neuron * (l + 1)] * r_pre
                W_a = torch.vstack([W_a, W_l])
                r_a = torch.vstack([r_a, r_l])
                W_i = torch.vstack([W_i, W_pre - W_l])
                r_i = torch.vstack([r_i, -torch.reshape(r_pre, [len(r_pre), 1]) + r_l])
            B_act = [W_a, r_a]  # W_a x <= r_a
            B_inact = [W_i, r_i]  # W_a x <= r_a
        # W_overl = torch.matmul(W_list[-1], torch.matmul(W_list[-2], W_l))  # compute \overline{W}(S)
        # r_overl = torch.matmul(W_list[-1], torch.matmul(W_list[-2], r_l) + r_list[-2].reshape([num_neuron,1])) + r_list[-1]  # compute \overline{r}(S)
        W_overl = torch.matmul(W_list[-1], W_l)  # compute \overline{W}(S)
        r_overl = torch.matmul(W_list[-1], r_l) + r_list[-1]  # compute \overline{r}(S)
        return W_overl, r_overl, B_act, B_inact

    def find_intersects(self, actuatl_set_list, possible_intersections):
        intersections = []
        act_intersections_list = []
        actuatl_set = np.asarray(actuatl_set_list)
        for sets in possible_intersections:
            cnt = 0
            set_asarray = np.asarray(sets.copy())
            set_temp = []
            act_set_temp = []
            for item in sets:
                if item in actuatl_set:
                    cnt += 1
                    act_str = np.array2string(item.reshape([len(item)]))
                    set_temp.append(act_str)
                    act_set_temp.append(item)
            additional_set = set(set_temp)
            if cnt > 1 and additional_set not in intersections:
                intersections.append(set(set_temp))
                act_intersections_list.append(act_set_temp.copy())
        return intersections, act_intersections_list

    def preceed_decompose(self):
        activated_sets, sections = self.list_activated()
        l1z = self.layers[1]
        l1a = self.layers[2]
        l2z = self.layers[3]
        l2a = self.layers[4]

        act_sets_list = []
        U_actset_list = []
        possible_intersections = []
        for item in range(len(sections)):
            [out_w, out_a, activated] = self.output_forward_activation(sections[item].reshape([2]), l1z, l1a)
            act_array = activated.int().numpy()
            [out_w, out_a, activated] = self.output_forward_activation(sections[item].reshape([2]), l2z, l2a)
            act_array = np.vstack([act_array, activated])
            act_str = np.array2string(act_array.reshape([len(act_array)]))
            U_actset_list.append(act_str)
            act_sets_list.append(act_array)
            if len(activated_sets[item]) > 0:
                # print(activated_sets[item])
                lst = list(itertools.product([0, 1], repeat=len(activated_sets[item])))
                intersect_items = []
                for possible in lst:
                    act_array[activated_sets[item]] = np.array(possible).reshape([len(possible), 1, 1])
                    act_str = np.array2string(act_array.reshape([len(act_array)]))
                    U_actset_list.append(act_str)
                    act_sets_list.append(act_array.copy())
                    intersect_items.append(act_array.copy())
                possible_intersections.append(intersect_items)
        return act_sets_list, activated_sets, possible_intersections

