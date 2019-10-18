# Neural Net implementation using PyTorch (data can include categorical data)
# For more details visit: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# Architectural details inspired from Denoising images : https://arxiv.org/abs/1705.02737

import numpy as np
import torch
import torch.nn as nn


class DenoisingAutoEncoder(nn.Module):
    """
    TO DO:
    -: Should the last layer be activated? #Check convergence details
    -: Should Batch Normalization be added?
    """

    def __init__(self, num_variables, one_hot_encoded_indexes, theta=7, input_dropout=0.5, dropout_at_layers = [0], logger_level=20):
        """
        dropout_at_layers => The layers of the NN where dropout needs to be added
        dropout_at_layers = [0] -> adds dropout to only the input layer
        dropout_at_layers = [0, 1] -> adds dropout to input and first layer etc..
        
        """
        import logging
        logger = logging.getLogger()
        logger.setLevel(logger_level)
        
        super().__init__()
        
        self.n = num_variables  # n will be the number of input features to the first layer
        logging.debug("self.n "+str(self.n))
        self.one_hot_encoded_indexes = one_hot_encoded_indexes
        
        self.drop_layer = nn.Dropout(p=input_dropout)  # Stochastic input distortion (50%)
        self.dropout_at_layers = list(set(sorted(dropout_at_layers))) #remove duplicates, ascending in order 
        assert set(self.dropout_at_layers).issubset([0,1,2,3,4,5]),"Allowed => [0,1,2,3,4,5]"
        
        units_per_layer = [self.n + i for i in (0, theta, theta * 2, theta * 3, theta * 2, theta * 1, 0)]
        zip_list = list(zip(units_per_layer, units_per_layer[1:]))
        # [(100, 107), (107, 114), (114, 121), (121, 114), (114, 107), (107, 100)] for n=100

        self.linear_layer_list = nn.ModuleList()
        for in_out_pair in zip_list:
            self.linear_layer_list.append(nn.Linear(*in_out_pair))

    def forward(self, x):
        x = x.float()
        h = x
        if 0 in self.dropout_at_layers: #Dropout at input layer
            h = self.drop_layer(x)
            
        for index, layer in enumerate(self.linear_layer_list[:-1], start = 1):
            h = torch.tanh(layer(h))
            if index in self.dropout_at_layers: # Applying dropout after activation
                h = self.drop_layer(h)

        output = self.linear_layer_list[-1](h)
        
        softmax_layer = nn.Softmax(dim = 0)
        for (i,j) in self.one_hot_encoded_indexes:
            output[i:j] = softmax_layer(output[i:j].clone())
        
        return output


if __name__ == "__main__":
    np.random.seed(18)
    test = torch.rand(1,4,5)
    print("Test\n",test)
    net = DenoisingAutoEncoder(num_variables=5)
    print("Output")
    print(net(test))