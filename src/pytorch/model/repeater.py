import torch


class repeater_module(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * 10


repeater = repeater_module.apply
