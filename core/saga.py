import torch
from torch.optim.optimizer import Optimizer


class SAGA(Optimizer):
    def __init__(self, params, local_grad, global_grad, lr=1e-3):
        defaults = dict(lr=lr)
        self.local_grad = local_grad
        self.global_grad = global_grad
        super(SAGA, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self):
        for p_group in self.param_groups:
            for p, lg, gg in zip(p_group['params'], self.local_grad, self.global_grad):
                if p.grad is None:
                    continue
                d_p = p.grad

                p.add_(d_p - lg + gg, alpha=-p_group['lr'])
