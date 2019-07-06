# Neural Net implementation using PyTorch
# For more details visit: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# Architectural details visit : https://arxiv.org/abs/1705.02737

import numpy as np
import torch
import torch.nn as nn


class DenoisingAutoEncoder(nn.Module):
    """
    TO DO:
    -: Should the last layer be activated? #Check convergence details
    -: Should Batch Normalization be added?
    """

    def __init__(self, num_variables, theta=7, input_dropout=0.5, logger_level=10):
        super().__init__()
        self.n = num_variables  # n will be the number of input features to the first layer

        self.drop_layer = nn.Dropout(p=input_dropout)  # Stochastic input distortion (50%)
        units_per_layer = [self.n + i for i in (0, theta, theta * 2, theta * 3, theta * 2, theta * 1, 0)]
        zip_list = list(zip(units_per_layer, units_per_layer[1:]))
        # [(100, 107), (107, 114), (114, 121), (121, 114), (114, 107), (107, 100)] for n=100

        self.linear_layer_list = nn.ModuleList()
        for in_out_pair in zip_list:
            self.linear_layer_list.append(nn.Linear(*in_out_pair))

    def forward(self, x):
        x = x.float()
        h = self.drop_layer(x)
        for layer in self.linear_layer_list[:-1]:
            h = torch.tanh(layer(h))

        output = self.linear_layer_list[-1](h)
        return output


if __name__ == "__main__":
    np.random.seed(18)
    test = torch.rand(1,4,5)
    print("Test\n",test)
    net = DenoisingAutoEncoder(5)
    print("Output")
    print(net(test))

