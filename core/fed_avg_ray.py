import torch
import torch.optim as optim
import copy
import ray
from utils import get_step_size_scheme, average_functions

class Worker:
    def __init__(self, data, label, loss, local_steps=10, mb_size=500, device='cuda'):
        self.data = data
        self.label = label
        self.n_class = len(torch.unique(self.label))
        self.local_steps = local_steps
        self.loss = loss
        self.device = device
        self.mb_size = mb_size

    def local_sgd(self, f_global, lr_0):
        f_local = copy.deepcopy(f_global)
        f_local.requires_grad_(True)
        optimizer = optim.Adam(f_local.parameters(), lr=lr_0)
        for local_iter in range(self.local_steps):
            optimizer.zero_grad()
            index = torch.unique(torch.randint(low=0, high=self.data.shape[0], size=(self.mb_size, )))
            loss = self.loss(f_local(self.data[index]), self.label[index])
            loss.backward()
            optimizer.step()

        return f_local



class Server:
    def __init__(self, workers, init_model, step_size_0=1, local_steps=10, device='cuda', use_ray=True, n_ray_workers=2):
        self.n_workers = len(workers)
        self.workers = workers
        self.f = init_model
        self.n_round = 0
        self.step_size_0 = torch.tensor(step_size_0, dtype=torch.float32, device=device)
        self.local_steps = local_steps
        self.device = device
        self.n_ray_workers = n_ray_workers
        self.use_ray = use_ray
        if use_ray:
            assert type(self.n_ray_workers) is int and self.n_ray_workers > 0
            assert self.n_workers % self.n_ray_workers == 0
            ray.init()

    def global_step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, self.local_steps)
        if self.use_ray:
            workers_list = chunks(self.workers, self.n_ray_workers)
            f_new = []
            for workers in workers_list:
                f_new = f_new + ray.get(
                    [dispatch.remote(worker, self.f, step_size_scheme(0)) for worker in workers]
                )
            self.f = average_functions(f_new)
        else:
            self.f = average_functions([worker.local_sgd(self.f, step_size_scheme(0)) for worker in self.workers])
        self.n_round += 1

@ray.remote(num_gpus=0.5, max_calls=1)
def dispatch(worker, f, step_size_scheme):
    return worker.local_sgd(f, step_size_scheme)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]