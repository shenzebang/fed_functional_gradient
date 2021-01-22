import torch
from utils import FunctionEnsemble, average_function_ensembles, merge_function_ensembles, weak_oracle

class Worker:
    def __init__(self, data, label, Dx_loss, get_init_weak_learner, local_steps=10, oracle_steps=10,
                 oracle_step_size=0.1, use_residual=True, use_ray=False, device='cuda'):
        self.data = data
        self.label = label
        self.n_class = len(torch.unique(self.label))
        self.local_steps = local_steps
        self.Dx_loss = Dx_loss
        self.oracle_steps = oracle_steps
        self.oracle_step_size = oracle_step_size
        self.device = device
        self.get_init_weak_learner = get_init_weak_learner
        self.use_residual = use_residual
        self.memory = None

    def local_fgd(self, f_inc, step_size_scheme):
        # print(f"in @ {time.time()}")
        with torch.autograd.no_grad():
            f_data_inc = f_inc(self.data)
            if self.memory is None:
                f_data = f_data_inc
            else:
                f_data = f_data_inc + self.memory

        self.memory = f_data

        # print(torch.norm(f_whole(self.data) - self.memory).item()/torch.norm(self.memory))
        f_new = FunctionEnsemble(empty=True)
        if self.use_residual:
            residual = torch.zeros(self.data.shape[0], self.n_class, dtype=torch.float32, device=self.device)
        for local_iter in range(self.local_steps):
        # subgradient is Dx_loss(f(data), label)
            target = self.Dx_loss(f_data, self.label)
            target = target + residual if self.use_residual else target
            target = target.detach()
            g, residual, g_data = weak_oracle(target, self.data, self.oracle_step_size,
                                  self.oracle_steps, init_weak_learner=self.get_init_weak_learner())
            f_new.add_function(g, -step_size_scheme(local_iter))
            with torch.autograd.no_grad():
                f_data = f_data - step_size_scheme(local_iter) * g_data
        # print(f"out @ {time.time()}")

        return f_new

class Server:
    def __init__(self, workers, get_init_weak_leaner, step_size_0=1, local_steps=10, device='cuda', cross_device=False):
        self.n_workers = len(workers)
        self.workers = workers
        self.f = FunctionEnsemble(get_init_function=get_init_weak_leaner, device=device)  # random initialization
        self.n_round = 0
        self.step_size_0 = torch.tensor(step_size_0, dtype=torch.float32, device=device)
        self.local_steps = local_steps
        self.device = device
        self.f_new = self.f
        self.cross_device = cross_device

    def global_step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, self.local_steps)
        self.f_new = average_function_ensembles([worker.local_fgd(self.f_new, step_size_scheme) for worker in self.workers])
        self.f = merge_function_ensembles([self.f, self.f_new])
        self.n_round += 1