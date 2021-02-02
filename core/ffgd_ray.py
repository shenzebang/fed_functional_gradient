import torch
import ray
from utils import FunctionEnsemble, average_function_ensembles, merge_function_ensembles, weak_oracle, get_step_size_scheme
from tqdm import tqdm
class Worker:
    """
    This class is implemented using ray. The local memory is simulated using global memory.
    """
    def __init__(self, data, label, Dx_loss, get_init_weak_learner, n_class, local_steps=10, oracle_steps=10,
                 oracle_step_size=0.1, mb_size=500, use_residual=True, device='cuda'):
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

    def local_fgd(self, memory, f_inc, step_size_scheme):
        # print(f"in @ {time.time()}")
        with torch.autograd.no_grad():
            f_data_inc = f_inc(self.data)
            if memory is None:
                f_data = f_data_inc
            else:
                f_data = f_data_inc + memory

        memory = f_data

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
                                  self.oracle_steps, init_weak_learner=self.get_init_weak_learner(), mb_size=self.mb_size)
            f_new.add_function(g, -step_size_scheme(local_iter))
            with torch.autograd.no_grad():
                f_data = f_data - step_size_scheme(local_iter) * g_data
        # print(f"out @ {time.time()}")

        return f_new, memory

# The function ensemble should be on cpu to save gpu memory
class Server:
    def __init__(self, workers, get_init_weak_leaner, step_size_0=1, local_steps=10, use_ray=True,
                 n_ray_workers=2, device='cuda', cross_device=False, store_f=True, step_size_decay_p=1):
        self.n_workers = len(workers)
        self.local_memories = [None]*self.n_workers
        self.workers = workers
        # random initialization & f should be on cpu to save gpu memory
        self.f = FunctionEnsemble(get_init_function=get_init_weak_leaner, device=device)
        self.n_round = 0
        self.step_size_0 = torch.tensor(step_size_0, dtype=torch.float32, device=device)
        self.local_steps = local_steps
        self.device = device
        self.f_new = self.f.to(device)
        self.use_ray = use_ray
        self.n_ray_workers = n_ray_workers
        self.step_size_decay_p = step_size_decay_p
        # ray does not work now
        if self.use_ray:
            ray.init()
            assert type(self.n_ray_workers) is int and self.n_ray_workers > 0
            assert self.n_workers % self.n_ray_workers == 0

        self.cross_device = cross_device
        self.store_f = store_f

    def global_step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, self.local_steps, self.step_size_decay_p)
        if self.use_ray:
            # workers = list(range(0, self.n_workers))
            workers_list = chunks(self.workers, self.n_ray_workers)
            memories_list = chunks(self.local_memories, self.n_ray_workers)
            results = []
            for workers, memories in tqdm(zip(workers_list, memories_list)):
                results = results + ray.get(
                    [dispatch.remote(worker, memory, self.f_new, step_size_scheme) for worker, memory in zip(workers, memories)]
                )
            # results is a list of tuples, each tuple is (f_new, memory) from a worker
            f_new = []
            memory = []
            for result in results:
                f_new.append(result[0])
                memory.append(result[1])
            self.local_memories = memory
            self.f_new = average_function_ensembles(f_new)
        else:
            self.f_new = average_function_ensembles([worker.local_fgd(self.f_new, step_size_scheme) for worker in self.workers])
        if self.store_f:
            # why store f if f is not to be used anyway.
            self.f = merge_function_ensembles([self.f, self.f_new])

        # self.f = merge_function_ensembles([self.f, self.f_new.to("cpu")])
        self.n_round += 1

@ray.remote(num_gpus=0.5, max_calls=1)
def dispatch(worker, memory, f_new, step_size_scheme):
    # print("dispatch")
    return worker.local_fgd(memory, f_new, step_size_scheme)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]