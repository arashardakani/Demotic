import torch

def AND(*args):
    return torch.prod(torch.stack(args, dim = 1).squeeze(1), dim = 1)

def NAND(*args):
    return 1 - torch.prod(torch.stack(args, dim = 1).squeeze(1), dim = 1)

def OR(*args):
    return 1 - torch.prod(1 - torch.stack(args, dim = 1).squeeze(1), dim = 1)

def NOR(*args):
    return torch.prod(1 - torch.stack(args, dim = 1).squeeze(1), dim = 1)

def XOR(a, b):
    return 1 - (1 - a * (1 - b)) * (1 - (1 - a) * b) 

def XNOR(a, b):
    return 1 - (1 - (1 - a) * (1 - b)) * (1 - a * b) 

def NOT(a):
    return 1 - a

def BUF(a):
    return a