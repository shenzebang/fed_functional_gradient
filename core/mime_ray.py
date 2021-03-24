import torch
import torch.optim as optim
import copy
import ray

from utils import get_step_size_scheme, average_functions, average_grad, chunks
from core.sgdm import SGDm

class Worker:
    def __init__(self, data, label, loss, local_steps=10, beta=0.9, mb_size=500, device='cuda'):
        self.data = data
        self.label = label
        self.n_class = len(torch.unique(self.label))
        self.local_steps = local_steps
        self.loss = loss
        self.device = device
        self.mb_size = mb_size
        self.beta = beta

    def init_local_grad(self, f):
        loss = self.loss(f(self.data), self.label)
        # self.local_grad =
        return torch.autograd.grad(loss, f.parameters())

    def local_sgd(self, f_global, momemtum, lr_0):
        f_local = copy.deepcopy(f_global)
        f_local.requires_grad_(True)

        optimizer = SGDm(f_local.parameters(), momemtum=momemtum, beta=self.beta, lr=lr_0)
        for local_iter in range(self.local_steps):
            optimizer.zero_grad()
            if 0 < self.mb_size < self.data.shape[0]:
                index = torch.unique(torch.randint(low=0, high=self.data.shape[0], size=(self.mb_size*2, )))
                index = index[:self.mb_size]
                loss = self.loss(f_local(self.data[index]), self.label[index])
            else:
                loss = self.loss(f_local(self.data), self.label)
            loss.backward()
            optimizer.step()

        loss = self.loss(f_global(self.data), self.label)
        local_grad = torch.autograd.grad(loss, f_global.parameters())
        return f_local, local_grad



class Server:
    def __init__(self, workers, init_model, step_size_0=1, local_steps=10, device='cuda', p=.1, beta=0.9, n_ray_workers=1):
        self.n_workers = len(workers)
        self.workers = workers
        self.f = init_model
        self.f.requires_grad_(True)
        self.n_round = 0
        self.step_size_0 = torch.tensor(step_size_0, dtype=torch.float32, device=device)
        self.local_steps = local_steps
        self.device = device
        self.momentum = self.init_and_aggr_local_grad()
        self.beta = beta
        self.p = p

        # initializing ray
        self.n_ray_workers = n_ray_workers
        assert type(self.n_ray_workers) is int and self.n_ray_workers > 0
        assert self.n_workers % self.n_ray_workers == 0
        ray.init()

    def global_step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, self.local_steps, p=self.p)

        workers_list = chunks(self.workers, self.n_ray_workers)
        results = []
        for workers in workers_list:
            results = results + ray.get(
                [dispatch.remote(worker, self.f, self.momentum, step_size_scheme(0))
                 for worker in workers]
            )

        # results = [worker.local_sgd(self.f, self.momentum, step_size_scheme(0)) for worker in self.workers]
        self.f = average_functions([result[0] for result in results])
        update_momentum(self.momentum, self.beta, average_grad([result[1] for result in results]))
        self.n_round += 1

    def init_and_aggr_local_grad(self):
        local_grads = [worker.init_local_grad(self.f) for worker in self.workers]
        global_grad = average_grad(local_grads)
        return global_grad


def update_momentum(momentum, beta, average_grad):
    # momentum = beta * momentum + (1-beta)*average_grad
    for m, g in zip(momentum, average_grad):
        m.mul_(beta)
        m.add_((1-beta)*g)

@ray.remote(num_gpus=1)
def dispatch(worker, f, momentum, step_size_scheme):
    return worker.local_sgd(f, momentum, step_size_scheme)