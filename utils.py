import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as vF




DATASETS = {
    "cifar": datasets.CIFAR10,
    "mnist": datasets.MNIST,
    "emnist": datasets.EMNIST
}


def get_flat_grad_from(grad):
    flat_grad = torch.cat([torch.flatten(p) for p in grad])
    return flat_grad


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model:
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def average_grad(grads):
    # flatten the grads to tensors
    flat_grads = []
    for grad in grads:
        flat_grads.append(get_flat_grad_from(grad))

    average_flat_grad = torch.mean(torch.stack(flat_grads), dim=0)
    grad_0 = grads[0]
    average_grad_a = []
    for p in grad_0:
        average_grad_a.append(torch.zeros_like(p))

    set_flat_params_to(average_grad_a, average_flat_grad)

    # average_grad_b = []
    # for i, p in enumerate(grad_0):
    #     flat_average_grad_p = torch.mean(torch.stack([torch.flatten(grad[i]) for grad in grads]), dim=0)
    #     average_grad_b.append(flat_average_grad_p.view_as(p))

    return average_grad_a


def average_functions(models):
    average_model = models[0]
    sds = [model.state_dict() for model in models]
    average_sd = sds[0]
    for key in sds[0]:
        average_sd[key] = torch.mean(torch.stack([sd[key] for sd in sds]), dim=0)
    average_model.load_state_dict(average_sd)
    return average_model


def average_function_ensembles(function_ensembles):
    new_function_ensemble = FunctionEnsemble(empty=True, device=function_ensembles[0].device)
    n_ensembles = len(function_ensembles)
    for ensemble in function_ensembles:
        new_function_ensemble.add_ensemble(ensemble)
    new_function_ensemble.rescale_weights(1. / n_ensembles)
    return new_function_ensemble


def merge_function_ensembles(function_ensembles):
    new_function_ensemble = FunctionEnsemble(empty=True, device=function_ensembles[0].device)
    for ensemble in function_ensembles:
        for function, weight in zip(ensemble.function_list, ensemble.weight_list):
            new_function_ensemble.add_function(function, weight)
    return new_function_ensemble


class FunctionEnsemble(nn.Module):
    def __init__(self, get_init_function=None, device='cuda', empty=False):
        super(FunctionEnsemble, self).__init__()
        if not empty and get_init_function is None:
            raise NotImplementedError
        self.function_list = [] if empty is True else [get_init_function().to(device)]
        self.weight_list = [] if empty is True else [torch.tensor([0.0], device=device)]
        self.device = device

    def forward(self, x):
        y = torch.tensor((0.0), device=self.device)
        for (function, weight) in zip(self.function_list, self.weight_list):
            y = y + weight * function(x)

        return y

    def add_function(self, f, weight):
        self.function_list.append(f)
        self.weight_list.append(weight)

    def add_ensemble(self, ensemble):
        self.function_list = self.function_list + ensemble.function_list
        self.weight_list = self.weight_list + ensemble.weight_list

    def rescale_weights(self, factor):
        self.weight_list = [weight * factor for weight in self.weight_list]

    def is_empty(self):
        return True if len(self.function_list) == 0 else False

    def to(self, device):
        new_ensemble = FunctionEnsemble(device=device, empty=True)
        new_ensemble.function_list = [function.to(device) for function in self.function_list]
        new_ensemble.weight_list = [weight.to(device) for weight in self.weight_list]
        return new_ensemble


class WeakLearnerConv(nn.Module):
    def __init__(self, height, width, n_class=10, n_channels=3, hidden_size=(384, 192), device='cuda'):
        # super(WeakLearnerConv, self).__init__()
        # self.device = device
        # assert (width == 32 and height == 32)
        # self.activation = F.leaky_relu
        # self.conv1 = nn.Conv2d(n_channels, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, hidden_size[0])
        # self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        # self.fc3 = nn.Linear(hidden_size[1], n_class)
        #
        # self.requires_grad_(False)
        # self.to(device)
        # self.apply(weights_init)

        super(WeakLearnerConv, self).__init__()
        self.device = device
        assert (width == 32 and height == 32)
        self.activation = F.leaky_relu
        self.conv1 = nn.Conv2d(n_channels, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], n_class)

        self.requires_grad_(False)
        self.to(device)

    # def forward(self, x):
    #     x = self.pool(self.activation(self.conv1(x)))
    #     x = self.pool(self.activation(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = self.activation(self.fc1(x))
    #     x = self.activation(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(x.shape[0], -1)
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
        nn.init.normal_(self.fc1.weight.data, 0.0, 1 / self.width / self.height / self.n_channel)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        nn.init.normal_(self.fc2.weight.data, 0.0, 1 / hidden_size[0])
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_size[1], n_class)
        nn.init.normal_(self.fc3.weight.data, 0.0, 1 / hidden_size[1])
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
    # optimizer = optim.SGD(g.parameters(), lr=lr, momentum=.9, weight_decay=1.e-4)
    num_epochs = oracle_steps * mb_size // data.shape[0] + 1
    for epoch in range(num_epochs):
        shuffle_index = torch.randperm(data.shape[0])
        data, target = data[shuffle_index], target[shuffle_index]
        _p = 0
        while _p + mb_size <= data.shape[0]:
            optimizer.zero_grad()
            loss = MSEloss(target[_p: _p+mb_size], g(data[_p: _p+mb_size]))
            loss.backward()
            optimizer.step()
            _p += mb_size


    # print(MSEloss(target, g(data))/torch.norm(target))
    # print(torch.norm(target))
    g.requires_grad_(False)
    if data.shape[0] <= 5000:
        g_data = g(data)
    else: # make sure the the batch size is no more than 5000
        num_chunk = data.shape[0] // 5000 + 1
        g_data = torch.cat([g(_d) for _d in torch.chunk(data, num_chunk)])
    residual = target - g_data
    # print(torch.norm(residual)/torch.norm(target))
    # print(torch.norm(g_data)/torch.norm(target))
    # print(torch.norm(residual[index])/torch.norm(target[index]))
    # print(torch.norm(g_data[index])/torch.norm(target[index]))

    return g, residual, g_data


def distill_oracle(target, data, lr, oracle_steps, init_weak_learner, mb_size=500):
    g = init_weak_learner
    MSEloss = nn.MSELoss()

    target = target.detach()
    g.requires_grad_(True)
    # index = torch.tensor([0, 1, 2, 3, 4, 5])
    optimizer = optim.Adam(g.parameters(), lr=lr)
    # optimizer = optim.SGD(g.parameters(), lr=lr, momentum=.9, weight_decay=1.e-4)
    num_epochs = oracle_steps * mb_size // data.shape[0] + 1
    for epoch in range(num_epochs):
        shuffle_index = torch.randperm(data.shape[0])
        data, target = data[shuffle_index], target[shuffle_index]
        _p = 0
        while _p + mb_size <= data.shape[0]:
            optimizer.zero_grad()
            loss = MSEloss(target[_p: _p+mb_size], g(data[_p: _p+mb_size]))
            loss.backward()
            optimizer.step()
            _p += mb_size

    del loss
    resi_norm = torch.mean(torch.sum((target - g(data))**2, dim=(1,)))
    print(f"residual norm of the distillation oracle {resi_norm}")
    return g

class FWeakOracle:
    def __init__(self, workers, n_rounds=20, local_sgd_step_size=1e-3):
        self.n_rounds = n_rounds  # number of communication rounds for every federated oracle query
        self.workers = workers
        self.n_workers = len(workers)
        self.local_sgd_step_size = local_sgd_step_size

    def query(self, f_global, init_weak_learner):
        g = init_weak_learner
        for worker in self.workers:
            worker.compute_subgradient(f_global)

        for i in range(self.n_rounds):
            lr = self.local_sgd_step_size / (0.1 * i + 1)
            g = average_functions([worker.local_sgd(g, lr) for worker in self.workers])

        for worker in self.workers:
            worker.update_residual(g)

        print(torch.max(torch.stack([torch.norm(worker.residual) for worker in self.workers])).item())

        return g


def get_step_size_scheme(n_round, step_size_0, local_steps, p=1):
    def step_size_scheme(k):
        # return step_size_0/((n_round+1) * local_steps + k + 1)**p
        return step_size_0 / ((n_round + 1) * local_steps + k + 1) ** p

    return step_size_scheme


def get_init_weak_learner(height, width, n_channel, n_class, hidden_size, type, device='cuda'):
    if type == "MLP":
        return WeakLearnerMLP(height=height, width=width, n_class=n_class, hidden_size=hidden_size, device=device)
    elif type == "Conv":
        return WeakLearnerConv(height=height, width=width, n_class=n_class, n_channels=n_channel,
                               hidden_size=hidden_size, device=device)
    else:
        raise NotImplementedError("Unknown weak learner type")


def data_partition(data, label, n_workers, homo_ratio):
    '''

    :param data:
    :param label:
    :param n_workers:
    :param homo_ratio: the portion of data to be allocated iid among $n_workers$ workers
    :param n_augment: augment the data set
    :return: lists of chunked data and labels
    '''

    if n_workers == 1:
        return [data], [label]

    assert data.shape[0] == label.shape[0]

    if n_workers == 1:
        return [data], [label]

    n_data = data.shape[0]

    n_homo_data = int(n_data * homo_ratio)

    n_homo_data = n_homo_data - n_homo_data % n_workers
    n_data = n_data - n_data % n_workers

    if n_homo_data > 0:
        data_homo, label_homo = data[0:n_homo_data], label[0:n_homo_data]
        data_homo_list, label_homo_list = data_homo.chunk(n_workers), label_homo.chunk(n_workers)

    if n_homo_data < n_data:
        data_hetero, label_hetero = data[n_homo_data:n_data], label[n_homo_data:n_data]
        label_hetero_sorted, index = torch.sort(label_hetero)
        data_hetero_sorted = data_hetero[index]

        data_hetero_list, label_hetero_list = data_hetero_sorted.chunk(n_workers), label_hetero_sorted.chunk(n_workers)

    if 0 < n_homo_data < n_data:
        data_list = [torch.cat([data_homo, data_hetero], dim=0) for data_homo, data_hetero in
                     zip(data_homo_list, data_hetero_list)]
        label_list = [torch.cat([label_homo, label_hetero], dim=0) for label_homo, label_hetero in
                      zip(label_homo_list, label_hetero_list)]
    elif n_homo_data < n_data:
        data_list = data_hetero_list
        label_list = label_hetero_list
    else:
        data_list = data_homo_list
        label_list = label_homo_list

    return data_list, label_list

def make_dataloaders(data_list, label_list, transforms=None):
    dss = make_datasets(data_list, label_list, transforms)
    dataloaders = [DataLoader(ds, batch_size=64, shuffle=True, num_workers=4) for ds in dss]
    return dataloaders


def make_adv_label(label_list, n_classes):
    # For machine i, define r = i%n. All data points on machine i that has label r is placed with label (r+1)%n_classes
    # n_workers = len(label_list)
    n_changed = 0
    n_total = 0
    new_label_list = []
    for i, label in enumerate(label_list):
        n_total += label.shape[0]
        r1 = i % n_classes
        index_1 = (label == r1).nonzero()
        n_changed += index_1.shape[0]
        r2 = (i + 1) % n_classes
        index_2 = (label == r2).nonzero()
        n_changed += index_2.shape[0]
        r1_new = (r1 + 1) % n_classes
        r2_new = (r2 + 1) % n_classes
        label[index_1] = r1_new
        label[index_2] = r2_new
        new_label_list.append(label)

    print(n_changed / n_total)
    return new_label_list


def load_data(args, hidden_size, device, augment_data=False):
    dataset_handle = DATASETS[args.dataset]

    if args.dataset == "mnist":
        dataset = dataset_handle(root='datasets/' + args.dataset, download=True)
        dataset_test = dataset_handle(root='datasets/' + args.dataset, train=False, download=True)
        data, label = dataset.train_data.to(dtype=torch.float32, device=device) / 255.0, \
                      dataset.train_labels.to(device=device)
        data = (data - torch.tensor(0.1307, device=device)) / torch.tensor(0.3081, device=device)
        assert (data.shape[0] == label.shape[0])
        (n_data, height, width) = data.shape
        n_class = 10

        if augment_data:
            data_h = vF.hflip(data)
            data = torch.cat([data, data_h], dim=0)
            label = torch.cat([label, label], dim=0)

        n_channel = 1
        rand_index = torch.randperm(data.shape[0])
        data, label = data[rand_index], label[rand_index]
        get_init_function = lambda: get_init_weak_learner(height, width, n_channel, n_class,
                                                          hidden_size=hidden_size, type="MLP", device=device)

        data_test, label_test = dataset_test.train_data.to(dtype=torch.float32, device=device) / 255.0, \
                                dataset_test.train_labels.to(device=device)
        data_test = (data_test - torch.tensor(0.1307, device=device)) / torch.tensor(0.3081, device=device)
        assert (data_test.shape[0] == label_test.shape[0])
        del rand_index

    elif args.dataset == "emnist":
        dataset = dataset_handle(root='datasets/' + args.dataset, split="byclass", download=True)
        dataset_test = dataset_handle(root='datasets/' + args.dataset, train=False, split="byclass", download=True)
        data, label = dataset.train_data.to(dtype=torch.float32, device=device) / 255.0, \
                      dataset.train_labels.to(device=device)
        data = (data - torch.tensor(0.1307, device=device)) / torch.tensor(0.3081, device=device)
        assert (data.shape[0] == label.shape[0])
        (n_data, height, width) = data.shape
        n_class = torch.unique(label).shape[0]

        if augment_data:
            data_h = vF.hflip(data)
            data = torch.cat([data, data_h], dim=0)
            label = torch.cat([label, label], dim=0)


        n_channel = 1
        rand_index = torch.randperm(data.shape[0])
        data, label = data[rand_index], label[rand_index]
        get_init_function = lambda: get_init_weak_learner(height, width, n_channel, n_class,
                                                          hidden_size=hidden_size, type="MLP", device=device)

        data_test, label_test = dataset_test.train_data.to(dtype=torch.float32, device=device) / 255.0, \
                                dataset_test.train_labels.to(device=device)
        data_test = (data_test - torch.tensor(0.1307, device=device)) / torch.tensor(0.3081, device=device)
        assert (data_test.shape[0] == label_test.shape[0])
        del rand_index

    elif args.dataset == "cifar":
        dataset = dataset_handle(root='datasets/' + args.dataset, download=True)
        dataset_test = dataset_handle(root='datasets/' + args.dataset, train=False, download=True)
        # processing training data
        data, label = torch.tensor(dataset.data, dtype=torch.float32, device=device) / 255.0, \
                      torch.tensor(dataset.targets, device=device)
        # normalize
        data = (data - torch.tensor((0.4914, 0.4822, 0.4465), device=device)) / torch.tensor((0.2023, 0.1994, 0.2010), device=device)
        data = data.permute(0, 3, 1, 2)  # from (H, W, C) to (C, H, W)

        # processing testing data
        data_test, label_test = torch.tensor(dataset_test.data, dtype=torch.float32, device=device) / 255.0, \
                                torch.tensor(dataset_test.targets, device=device)
        # normalize
        data_test = (data_test - torch.tensor((0.4914, 0.4822, 0.4465), device=device)) / torch.tensor((0.2023, 0.1994, 0.2010),
                                                                                              device=device)
        data_test = data_test.permute(0, 3, 1, 2)  # from (H, W, C) to (C, H, W)

        assert (data.shape[0] == label.shape[0])
        (n_data, n_channel, height, width) = data.shape
        n_class = 10
        if augment_data:
            data_h = vF.hflip(data)
            data = torch.cat([data, data_h], dim=0)
            label = torch.cat([label, label], dim=0)


        rand_index = torch.randperm(data.shape[0])
        data, label = data[rand_index], label[rand_index]
        get_init_function = lambda: get_init_weak_learner(height, width, n_channel, n_class,
                                                          hidden_size=hidden_size, type="Conv", device=device)

        del rand_index
    else:
        raise NotImplementedError

    return data, label, data_test, label_test, n_class, get_init_function


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:min(len(lst), i + n)]


def is_nan(x):
    return x != x


def make_dataset(data, label, transform):
    class LocalDataset(Dataset):
        def __init__(self, data, label, transform=None):
            self.data = data
            self.label = label
            self.transform = transform
            assert data.shape[0] == label.shape[0]

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, item):
            sample = self.data[item]
            if self.transform:
                sample = self.transform(sample)
            return sample, self.label[item]

    return LocalDataset(data, label)


def make_datasets(data_list, label_list, transform=None):
    return [make_dataset(data, label, transform) for data, label in zip(data_list, label_list)]
