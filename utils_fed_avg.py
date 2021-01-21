import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

def average_functions(models):
    average_model = models[0]
    sds = [model.state_dict() for model in models]
    average_sd = sds[0]
    for key in sds[0]:
        average_sd[key] = torch.mean(torch.stack([sd[key] for sd in sds]), dim=0)
    average_model.load_state_dict(average_sd)
    return average_model


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


class Worker:
    def __init__(self, data, label, loss, local_steps=10, device='cuda'):
        self.data = data
        self.label = label
        self.n_class = len(torch.unique(self.label))
        self.local_steps = local_steps
        self.loss = loss
        self.device = device

    def local_fgd(self, f_global, lr_0):
        f_local = copy.deepcopy(f_global)
        f_local.requires_grad_(True)
        optimizer = optim.Adam(f_local.parameters(), lr=lr_0)
        for local_iter in range(self.local_steps):
            optimizer.zero_grad()
            mb_size = 500
            index = torch.unique(torch.randint(low=0, high=self.data.shape[0], size=(mb_size, )))
            loss = self.loss(f_local(self.data[index]), self.label[index])
            loss.backward()
            optimizer.step()

        return f_local


def get_step_size_scheme(n_round, step_size_0, local_steps):
    def step_size_scheme(k):
        return step_size_0/((n_round+1) * local_steps + k + 1)

    return step_size_scheme


class Server:
    def __init__(self, workers, init_model, step_size_0=1, local_steps=10, device='cuda'):
        self.n_workers = len(workers)
        self.workers = workers
        self.f = init_model
        self.n_round = 0
        self.step_size_0 = torch.tensor(step_size_0, dtype=torch.float32, device=device)
        self.local_steps = local_steps
        self.device = device

    def global_step(self):
        step_size_scheme = get_step_size_scheme(self.n_round, self.step_size_0, self.local_steps)
        self.f = average_functions([worker.local_fgd(self.f, step_size_scheme(0)) for worker in self.workers])
        self.n_round += 1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)

def get_init_model(height, width, n_channel, n_class, hidden_size, type, device='cuda'):
    if type == "MLP":
        return WeakLearnerMLP(height=height, width=width, n_class=n_class, hidden_size=hidden_size, device=device)
    elif type == "Conv":
        return WeakLearnerConv(height=height, width=width, n_class=n_class, n_channels=n_channel, hidden_size=hidden_size, device=device)
    else:
        raise NotImplementedError("Unknown weak learner type")
