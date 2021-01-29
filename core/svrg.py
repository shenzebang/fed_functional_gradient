import torch
from torch.optim.optimizer import Optimizer


class SVRG(Optimizer):
    def __init__(self, params, full_grad, lr=1e-3):
        defaults = dict(lr=lr)
        # self.local_grad = local_grad
        self.full_grad = full_grad
        super(SVRG, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, local_grad):
        for p_group in self.param_groups:
            for p, lg, gg in zip(p_group['params'], local_grad, self.full_grad):
                if p.grad is None:
                    continue
                d_p = p.grad

                p.add_(d_p - lg + gg, alpha=-p_group['lr'])
