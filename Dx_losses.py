import torch
import torch.nn as nn


def Dx_cross_entropy(input, target):
    input.requires_grad = True
    loss = torch.nn.functional.cross_entropy(input, target, reduction='sum')
    grad = torch.autograd.grad(loss, input)[0]
    return grad