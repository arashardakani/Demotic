import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import gc
from .pgates import *



class PIEmbedding(nn.Module):
    def __init__(
        self,
        input_names: list[str],
        input_shape: list[int],
        device: str = "cpu",
        batch_size: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.device = device
        self.batch_size = batch_size
        self.input_names = input_names


        self.parameters_list = nn.ParameterList()
        for name, size in zip(input_names, input_shape):
            param = nn.Parameter(torch.randn(batch_size, size, device=device))
            self.parameters_list.append(param)

        self.activation = torch.nn.Sigmoid()  

    def forward(self):
        outputs = []
        for param in self.parameters_list:
            param.data.clamp_(-2., 2.)
            output_tensor = self.activation(4 * param)
            outputs.append(output_tensor)
        return outputs




class CircuitModel(nn.Module):
    """Combinational Circuit instantiated from a PySAT CNF problem"""

    def __init__(self, **kwargs):
        # read cnf file
        super().__init__()
        self.pytorch_model = kwargs["pytorch_model"]
        exec(self.pytorch_model)
        class_object = locals()[kwargs["module_name"]]
        self.num_clk_cycles = kwargs["num_clk_cycles"]

        
        self.emb = PIEmbedding(kwargs["inputs_str"], [1] * len(kwargs["inputs_str"]), kwargs["device"], kwargs["batch_size"])
        self.probabilistic_circuit_model = class_object(kwargs["batch_size"], kwargs["device"])
        

    def forward(self):
        x = self.emb()
        out = self.probabilistic_circuit_model(x)
        return out

