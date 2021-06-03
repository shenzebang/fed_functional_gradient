import torch
import ray
from utils import FunctionEnsemble, average_function_ensembles, weak_oracle, \
    get_step_size_scheme, distill_oracle

from torch.nn.functional import mse_loss
class Worker:
    """
    This class is implemented using ray. The local memory is simulated using global memory.
    """
    def __init__(self, data, label, Dx_loss, get_init_weak_learner, n_class, local_steps=10, oracle_steps=10,
                 oracle_step_size=0.1, mb_size=500, use_residual=True, device='cuda', n_distill_steps=4000, distill_step_size=5e-3):
        self.data = data
        self.label = label
        self.n_class = n_class
        self.local_steps = local_steps
        self.Dx_loss = Dx_loss
        self.oracle_steps = oracle_steps
        self.oracle_step_size = oracle_step_size
        self.device = device
        self.get_init_weak_learner = get_init_weak_learner
        self.use_residual = use_residual
        self.mb_size = mb_size
        self.n_distill_steps = n_distill_steps
        self.distill_step_size = distill_step_size

    def local_fgd(self, f, step_size_scheme):
        with torch.autograd.no_grad():
            f_data = f(self.data)

        f_inc = FunctionEnsemble(empty=True, device=self.device)
        if self.use_residual:
            residual = torch.zeros(self.data.shape[0], self.n_class, dtype=torch.float32, device=self.device)
        for local_iter in range(self.local_steps):
        # subgradient is Dx_loss(f(data), label)
            target = self.Dx_loss(f_data, self.label)
            target = target + residual if self.use_residual else target
            target = target.detach()
            g, residual, g_data = weak_oracle(target, self.data, self.oracle_step_size,
                                  self.oracle_steps, init_weak_learner=self.get_init_weak_learner(), mb_size=self.mb_size)
            f_inc.add_function(g, -step_size_scheme(local_iter))
            with torch.autograd.no_grad():
                f_data = f_data - step_size_scheme(local_iter) * g_data

        return f_inc, torch.mean(torch.sum(residual ** 2, dim=(1,)))


class Server:
    def __init__(self, workers, get_init_weak_leaner, x_distill, step_size_0=1, local_steps=10, use_ray=True,
                 n_ray_workers=2, device='cuda', step_size_decay_p=1):
        self.n_workers = len(workers)
        self.local_memories = [None]*self.n_workers
        self.workers = workers
        self.x_distill = x_distill
        # random initialization & f should be on cpu to save gpu memory
        self.f = FunctionEnsemble(get_init_function=get_init_weak_leaner, device=device)
        self.get_init_weak_leaner = get_init_weak_leaner
        self.n_round = 0
        self.step_size_0 = torch.tensor(step_size_0, dtype=torch.float32, device=device)
        self.local_steps = local_steps
        self.device = device
        self.use_ray = use_ray
        self.n_ray_workers = n_ray_workers
        self.step_size_decay_p = step_size_decay_p
        # ray does not work now
        if self.use_ray:
            # print(f"device is {device}")
            # print(device=="cpu")
            ray.init()

            if device.type == "cuda":
                self.dispatch = dispatch_cuda
            elif device.type == "cpu":
                self.dispatch = dispatch_cpu
            else:
                raise NotImplementedError

    def global_step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, self.local_steps, self.step_size_decay_p)
        print("====== client update ======")
        results = ray.get([self.dispatch.remote(worker, self.f, step_size_scheme) for worker in self.workers])
        f_new = [result[0] for result in results]
        res_norm = [result[1] for result in results]
        print(f"mean of res_norm is {torch.mean(torch.stack(res_norm))}, std of res_norm is {torch.std(torch.stack(res_norm))}")
        self.f.add_ensemble(average_function_ensembles(f_new))
        print("====== server update ======")
        f = FunctionEnsemble(empty=True)
        # distill_oracle(self.f(self.x_distill), self.x_distill, 1e-4, 10000, self.get_init_weak_leaner(), 128),
        # distill_oracle(self.f(self.x_distill), self.x_distill, 5e-4, 10000, self.get_init_weak_leaner(), 128),
        # distill_oracle(self.f(self.x_distill), self.x_distill, 1e-4, 20000, self.get_init_weak_leaner(), 128),
        # distill_oracle(self.f(self.x_distill), self.x_distill, 5e-4, 20000, self.get_init_weak_leaner(), 128),
        # f.add_function(
        #     distill_oracle(self.f(self.x_distill), self.x_distill, 1e-3, 10000, self.get_init_weak_leaner(), 128), 1.)
        f.add_function(
            distill_oracle(self.f(self.x_distill), self.x_distill, 5e-4, 20000, self.get_init_weak_leaner(), 128), 1.)
        self.f = f
        self.n_round += 1

@ray.remote(num_gpus=1, max_calls=1)
def dispatch_cuda(worker, f_new, step_size_scheme):
    return worker.local_fgd(f_new, step_size_scheme)

@ray.remote(num_cpus=1)
def dispatch_cpu(worker, f, step_size_scheme):
    return worker.local_fgd(f, step_size_scheme)
