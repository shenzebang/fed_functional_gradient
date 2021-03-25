import torch
import copy

from utils import get_step_size_scheme, average_functions, get_flat_grad_from
from torch.optim import SGD


class Worker:
    def __init__(self, data, label, loss, n_class, local_steps=10, mb_size=500, device='cuda', mu=1e-5):
        self.data = data
        self.label = label
        self.n_class = n_class
        self.local_steps = local_steps
        self.loss = loss
        self.device = device
        self.mb_size = mb_size
        self.mu = mu

    def local_sgd(self, f_global, lr_0):
        f_local = copy.deepcopy(f_global)
        f_local.requires_grad_(True)

        optimizer = SGD(f_local.parameters(), lr=lr_0)
        for local_iter in range(self.local_steps):
            optimizer.zero_grad()
            if 0 < self.mb_size < self.data.shape[0]:
                index = torch.unique(torch.randint(low=0, high=self.data.shape[0], size=(self.mb_size * 2,)))
                index = index[:self.mb_size]
                params_diff = get_flat_grad_from(f_local.parameters()) - get_flat_grad_from(
                    f_global.parameters()).detach()
                loss = self.loss(f_local(self.data[index]), self.label[index]) + \
                       self.mu * torch.sum(params_diff ** 2)
            else:
                loss = self.loss(f_local(self.data), self.label)
            loss.backward()
            optimizer.step()

        return f_local


class Server:
    def __init__(self, workers, init_model, step_size_0=1, local_steps=10, device='cuda', p=.1):
        self.n_workers = len(workers)
        self.workers = workers
        self.f = init_model
        self.f.requires_grad_(True)
        self.n_round = 0
        self.step_size_0 = torch.tensor(step_size_0, dtype=torch.float32, device=device)
        self.local_steps = local_steps
        self.device = device
        self.p = p

    def global_step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, self.local_steps, p=self.p)
        results = [worker.local_sgd(self.f, step_size_scheme(0)) for worker in self.workers]
        self.f = average_functions(results)
        self.n_round += 1
