import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ray
import time

def average_function_ensembles(function_ensembles):
    new_function_ensemble = FunctionEnsemble(empty=True)
    n_ensembles = len(function_ensembles)
    for ensemble in function_ensembles:
        new_function_ensemble.add_ensemble(ensemble)
    new_function_ensemble.rescale_weights(1./n_ensembles)
    return new_function_ensemble

def merge_function_ensembles(function_ensembles):
    new_function_ensemble = FunctionEnsemble(empty=True)
    for ensemble in function_ensembles:
        for function, weight in zip(ensemble.function_list, ensemble.weight_list):
            new_function_ensemble.add_function(function, weight)
    return new_function_ensemble


class FunctionEnsemble(nn.Module):
    def __init__(self, get_init_function=None, device='cuda', empty=False):
        super(FunctionEnsemble, self).__init__()
        if not empty and get_init_function is None:
            raise NotImplementedError
        self.function_list = [] if empty is True else [get_init_function()]
        self.weight_list = [] if empty is True else [torch.tensor([0.0], device=device)]
        self.device = device


    def forward(self, x):
        y = torch.tensor((0.0), device=self.device)
        for (function, weight) in zip(self.function_list, self.weight_list):
            y = y + weight*function(x)

        return y

    def add_function(self, f, weight):
        self.function_list.append(f)
        self.weight_list.append(weight)

    def add_ensemble(self, ensemble):
        self.function_list = self.function_list + ensemble.function_list
        self.weight_list = self.weight_list + ensemble.weight_list

    def rescale_weights(self, factor):
        self.weight_list = [weight*factor for weight in self.weight_list]

    def is_empty(self):
        return True if len(self.function_list) == 0 else False


class WeakLearnerConv(nn.Module):
    def __init__(self, height, width, n_class=10, n_channels=3, hidden_size=(128, 128), device='cuda'):
        super(WeakLearnerConv, self).__init__()
        self.device = device
        assert(width==32 and height==32)
        self.activation = F.leaky_relu
        self.conv1 = nn.Conv2d(n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], n_class)

        self.requires_grad_(False)
        self.to(device)
        # self.apply(weights_init)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class WeakLearnerMLP(nn.Module):
    def __init__(self, height, width, n_class, n_channel=1, hidden_size=(128, 128), device='cuda'):
        super(WeakLearnerMLP, self).__init__()
        self.device = device
        self.height = height
        self.width = width
        self.n_class = n_class
        self.n_channel = n_channel
        self.fc1 = nn.Linear(width * height * n_channel, hidden_size[0])
        nn.init.normal_(self.fc1.weight.data, 0.0, 1/self.width/self.height/self.n_channel)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        nn.init.normal_(self.fc2.weight.data, 0.0, 1/hidden_size[0])
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_size[1], n_class)
        nn.init.normal_(self.fc3.weight.data, 0.0, 1/hidden_size[1])
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        # self.dropout = nn.Dropout(0.2)
        self.activation = F.leaky_relu

        self.to(self.device)
        self.requires_grad_(False)
        # self.apply(weights_init)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, self.height * self.width * self.n_channel)
        # add hidden layer, with relu activation function
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x


def weak_oracle(target, data, lr, oracle_steps, init_weak_learner, mb_size=500):
    g = init_weak_learner
    MSEloss = nn.MSELoss()

    target = target.detach()
    # target = Dx_loss(f(data), label)
    # print(torch.norm(target)/(1e-8 + torch.norm(residual)))
    # Dx_loss(f(data), label) is the subgradient
    g.requires_grad_(True)
    # index = torch.tensor([0, 1, 2, 3, 4, 5])
    optimizer = optim.Adam(g.parameters(), lr=lr)

    for _ in range(oracle_steps):
        optimizer.zero_grad()
        index = torch.unique(torch.randint(low=0, high=data.shape[0], size=(mb_size, )))
        # g should approximate target on data
        # loss = torch.sum((target[rand_index] - g(data[rand_index])).pow(2))/mb_size
        # print(loss.item())
        loss = MSEloss(target[index], g(data[index]))
        loss.backward()
        optimizer.step()

    # print(MSEloss(target, g(data))/torch.norm(target))
    # print(torch.norm(target))
    g.requires_grad_(False)
    g_data = g(data)
    residual = target - g_data
    # print(torch.norm(residual)/torch.norm(target))
    # print(torch.norm(g_data)/torch.norm(target))
    # print(torch.norm(residual[index])/torch.norm(target[index]))
    # print(torch.norm(g_data[index])/torch.norm(target[index]))

    return g, residual, g_data

class Worker:
    """
    This class is implemented using ray. The local memory is simulated using global memory.
    """
    def __init__(self, data, label, Dx_loss, get_init_weak_learner, local_steps=10, oracle_steps=10,
                 oracle_step_size=0.1, mb_size=500, use_residual=True, device='cuda'):
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


def get_step_size_scheme(n_round, step_size_0, local_steps):
    def step_size_scheme(k):
        return step_size_0/((n_round+1) * local_steps + k + 1)

    return step_size_scheme


class Server:
    def __init__(self, workers, get_init_weak_leaner, step_size_0=1, local_steps=10, use_ray=True,
                 n_ray_workers=2, device='cuda', cross_device=False):
        self.n_workers = len(workers)
        self.local_memories = [None]*self.n_workers
        self.workers = workers
        self.f = FunctionEnsemble(get_init_function=get_init_weak_leaner, device=device)  # random initialization
        self.n_round = 0
        self.step_size_0 = torch.tensor(step_size_0, dtype=torch.float32, device=device)
        self.local_steps = local_steps
        self.device = device
        self.f_new = self.f
        self.use_ray = use_ray
        self.n_ray_workers = n_ray_workers
        # ray does not work now
        if self.use_ray:
            ray.init()
            assert type(self.n_ray_workers) is int and self.n_ray_workers > 0
            assert self.n_workers % self.n_ray_workers == 0

        self.cross_device = cross_device

    def global_step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, self.local_steps)
        if self.use_ray:
            # workers = list(range(0, self.n_workers))
            workers_list = chunks(self.workers, self.n_ray_workers)
            memories_list = chunks(self.local_memories, self.n_ray_workers)
            results = []
            for workers, memories in zip(workers_list, memories_list):
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
        self.f = merge_function_ensembles([self.f, self.f_new])
        self.n_round += 1

@ray.remote(num_gpus=0.5, max_calls=1)
def dispatch(worker, memory, f_new, step_size_scheme):
    # print("dispatch")
    return worker.local_fgd(memory, f_new, step_size_scheme)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)

def get_init_weak_learner(height, width, n_channel, n_class, hidden_size, type, device='cuda'):
    if type == "MLP":
        return WeakLearnerMLP(height=height, width=width, n_class=n_class, hidden_size=hidden_size, device=device)
    elif type == "Conv":
        return WeakLearnerConv(height=height, width=width, n_class=n_class, n_channels=n_channel, hidden_size=hidden_size, device=device)
    else:
        raise NotImplementedError("Unknown weak learner type")
