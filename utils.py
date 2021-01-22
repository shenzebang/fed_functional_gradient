import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


def weak_oracle(target, data, lr, oracle_steps, init_weak_learner):
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
        mb_size = 500
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


def get_step_size_scheme(n_round, step_size_0, local_steps):
    def step_size_scheme(k):
        return step_size_0/((n_round+1) * local_steps + k + 1)

    return step_size_scheme


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

def get_init_weak_learner(height, width, n_channel, n_class, hidden_size, type, device='cuda'):
    if type == "MLP":
        return WeakLearnerMLP(height=height, width=width, n_class=n_class, hidden_size=hidden_size, device=device)
    elif type == "Conv":
        return WeakLearnerConv(height=height, width=width, n_class=n_class, n_channels=n_channel, hidden_size=hidden_size, device=device)
    else:
        raise NotImplementedError("Unknown weak learner type")


def data_partition(data, label, n_workers, homo_ratio, n_augment=None):
    '''

    :param data:
    :param label:
    :param n_workers:
    :param homo_ratio: the portion of data to be allocated iid among $n_workers$ workers
    :param n_augment: augment the data set
    :return: lists of chunked data and labels
    '''
    if n_augment is not None:
        raise NotImplementedError

    assert data.shape[0] == label.shape[0]

    n_data = data.shape[0]

    homo_data = int(n_data * homo_ratio)

    data_homo, label_homo = data[0:homo_data], label[0:homo_data]
    data_homo_list, label_homo_list = data_homo.chunk(n_workers), label_homo.chunk(n_workers)

    data_hetero, label_hetero = data[homo_data:n_data], label[homo_data:n_data]
    label_hetero_sorted, index = torch.sort(label_hetero)
    data_hetero_sorted = data_hetero[index]

    data_hetero_list, label_hetero_list = data_hetero_sorted.chunk(n_workers), label_hetero_sorted.chunk(n_workers)

    data_list = [torch.cat([data_homo, data_hetero], dim=0) for data_homo, data_hetero in zip(data_homo_list, data_hetero_list)]
    label_list = [torch.cat([label_homo, label_hetero], dim=0) for label_homo, label_hetero in zip(label_homo_list, label_hetero_list)]

    return data_list, label_list