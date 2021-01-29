import torch
import torch.optim as optim
import copy

from utils import get_step_size_scheme, average_functions, average_grad
from core.svrg import SVRG

class Worker:
    def __init__(self, data, label, loss, local_steps=10, mb_size=500, device='cuda'):
        self.data = data
        self.label = label
        self.n_class = len(torch.unique(self.label))
        self.local_steps = local_steps
        self.loss = loss
        self.device = device
        self.mb_size = mb_size
        self.local_grad = None

    def init_local_grad(self, f):
        loss = self.loss(f(self.data), self.label)
        self.local_grad = torch.autograd.grad(loss, f.parameters())
        return self.local_grad

    def local_sgd(self, f_global, global_grad, lr_0):
        f_local = copy.deepcopy(f_global)
        f_local.requires_grad_(True)

        optimizer = SVRG(f_local.parameters(), full_grad=global_grad, lr=lr_0)
        for local_iter in range(self.local_steps):
            optimizer.zero_grad()
            if self.mb_size > 0:
                index = torch.unique(torch.randint(low=0, high=self.data.shape[0], size=(self.mb_size, )))
                data_index, label_index = self.data[index], self.label[index]
                loss_1 = self.loss(f_global(data_index), label_index)
                grad = torch.autograd.grad(loss_1, f_global.parameters())
                loss = self.loss(f_local(data_index), label_index)
            else:
                loss_1 = self.loss(f_global(self.data), self.label)
                grad = torch.autograd.grad(loss_1, f_global.parameters())
                loss = self.loss(f_local(self.data), self.label)
            loss.backward()
            optimizer.step(grad)

        loss = self.loss(f_global(self.data), self.label)
        self.local_grad = torch.autograd.grad(loss, f_global.parameters())
        return f_local, self.local_grad



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
        self.global_grad = self.init_and_aggr_local_grad()
        self.p = p

    def global_step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, self.local_steps, p=self.p)
        results = [worker.local_sgd(self.f, self.global_grad, step_size_scheme(0)) for worker in self.workers]
        self.f = average_functions([result[0] for result in results])
        self.global_grad = average_grad([result[1] for result in results])
        self.n_round += 1

    def init_and_aggr_local_grad(self):
        local_grads = [worker.init_local_grad(self.f) for worker in self.workers]
        global_grad = average_grad(local_grads)
        return global_grad