import torch
from torch.optim.optimizer import Optimizer


class SGDm(Optimizer):
    def __init__(self, params, momemtum, beta, lr=1e-3):
        defaults = dict(lr=lr)
        self.momemtum = momemtum
        self.beta = beta
        super(SGDm, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self):
        for p_group in self.param_groups:
            for p, m in zip(p_group['params'], self.momemtum):
                if p.grad is None:
                    continue
                d_p = p.grad

                p.add_((1-self.beta) * d_p + self.beta * m, alpha=-p_group['lr'])
