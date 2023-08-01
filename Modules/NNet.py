import torch
import torch.nn as nn
import numpy as np


class NNet:
    def __init__(self, arch, act_layer, DOMAIN, n_output=1, verbose=False):
        '''
        Initialize NCBF with a given architecture and ReLU layers
        :param arch: a vector of # of neurons in each layer
        :param act_layer: a vector of # of ReLU layers
        :param DOMAIN: state space domain
        '''
        assert len(arch) == len(act_layer), 'Arch should match act_layer'

        self.verbose = verbose
        self.arch = arch
        self.DOMAIN = DOMAIN
        self.act_fun = nn.Tanh()
        self.act_layer = act_layer
        self.device = self.get_device()
        self.DIM = len(self.DOMAIN)


        self.layer_input = [nn.Linear(len(DOMAIN), self.arch[0], bias=True)]
        self.layer_output = [self.act_fun, nn.Linear(self.arch[-1], n_output, bias=True)]

        # hidden layer
        self.module_hidden = []
        for i in range(len(arch) - 1):
            if self.act_layer[i]:
                self.module_hidden.append([self.act_fun, nn.Linear(self.arch[i], self.arch[i + 1], bias=True)])
            else:
                self.module_hidden.append([nn.Identity(), nn.Linear(self.arch[i], self.arch[i + 1], bias=True)])
        # self.module_hidden = [[self.act_fun, nn.Linear(self.arch[i], self.arch[i], bias=True)] for i in range(len(arch)-1)]
        self.layer_hidden = list(np.array(self.module_hidden).flatten())

        # nn model
        self.layers = self.layer_input + self.layer_hidden + self.layer_output
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

    def forward(self, x):
        return self.model.forward(x)

    def get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
            if self.verbose:
                print('Using cuda:0')
        else:
            device = 'cpu'
            if self.verbose:
                print('Using cpu')
        return device