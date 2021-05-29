import copy

import ray
import torch
from tqdm import tqdm
from utils import FunctionEnsemble, average_functions, get_step_size_scheme


class Server:
    def __init__(self, fed_oracle, get_init_weak_learner, step_size_0=1., p=1):
        self.fed_oracle = fed_oracle
        self.get_init_weak_learner = get_init_weak_learner
        self.n_round = 0
        self.step_size_0 = step_size_0
        self.p = p
        ray.init()

    def step(self):
        h_init = self.get_init_weak_learner()
        step_size = get_step_size_scheme(self.n_round, self.step_size_0, 1, self.p)(0)
        h = self.fed_oracle.step(h_init, step_size)
        self.n_round += 1

        f_inc = FunctionEnsemble(empty=True)
        f_inc.add_function(h, -step_size)
        return f_inc


class FedOracle:
    def __init__(self, x_list, y_list, Dx_loss, num_local_epoch, epoch_per_step, opt_lr, num_steps, num_classes=10):
        self.workers = [Worker(x, y, Dx_loss, num_local_epoch, epoch_per_step, opt_lr, num_classes) for
                        (x, y) in zip(x_list, y_list)]
        self.num_steps = num_steps

    def step(self, h, step_size):
        for step in range(self.num_steps):
            result = ray.get([dispatch_cuda.remote(worker, h) for worker in self.workers])
            h = average_functions(result)
        [worker.update_state(h, step_size) for worker in self.workers]
        return h


class Worker:
    def __init__(self, x, y, Dx_loss, num_local_epoch, epoch_per_step, opt_lr, num_classes=10):
        assert (x.shape[0] == y.shape[0])
        self.x = x
        self.y = y
        self.Dx_loss = Dx_loss
        self.f_x = torch.zeros((x.shape[0], num_classes), device="cuda")
        self.target = self.Dx_loss(self.f_x, self.y)
        self.num_local_epoch = num_local_epoch
        self.epoch_per_step = epoch_per_step
        self.opt_lr = opt_lr
        self.num_sample_per_step = int(self.epoch_per_step * self.x.shape[0])

    def step(self, h_init):
        h = copy.deepcopy(h_init)
        h.requires_grad_(True)

        mse = torch.nn.MSELoss()
        opt = torch.optim.SGD(h.parameters(), lr=self.opt_lr)
        for epoch in range(self.num_local_epoch):
            index = torch.randperm(self.x.shape[0])
            x = self.x[index]
            target = self.target[index]
            _p = 0
            while _p + self.num_sample_per_step <= self.x.shape[0]:
                opt.zero_grad()
                loss = mse(h(x[_p: _p + self.num_sample_per_step]), target[_p: _p + self.num_sample_per_step])
                loss.backward()
                opt.step()
                _p += self.num_sample_per_step

        return h

    def update_state(self, h, step_size):
        with torch.autograd.no_grad():
            self.f_x = self.f_x - h(self.x) * step_size
        self.target = self.Dx_loss(self.f_x, self.y).detach()

@ray.remote(num_gpus=1)
def dispatch_cuda(worker, h):
    return worker.step(h)
