import argparse
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from Dx_losses import Dx_cross_entropy
from tqdm import tqdm

import utils
import utils_ray
# from utils import Worker, Server
import numpy as np
import time

Dx_losses = {
    "logistic_regression": 123,
    "cross_entropy": Dx_cross_entropy
}
losses = {
    "logistic_regression": 123,
    "cross_entropy": lambda x, y: torch.nn.functional.cross_entropy(x, y, reduction='sum')
}
DATASETS = {
    "cifar": datasets.CIFAR10,
    "mnist": datasets.MNIST
}

if __name__ == '__main__':
    ts = time.time()
    algo = 'ffgd'
    parser = argparse.ArgumentParser(algo)

    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--weak_learner_hid_dims', type=str, default='32-32')
    parser.add_argument('--step_size_0', type=float, default=10.0)
    parser.add_argument('--loss', type=str, choices=['logistic_regression', 'l2_regression', 'cross_entropy'],
                        default='cross_entropy')
    parser.add_argument('--worker_local_steps', type=int, default=10)
    parser.add_argument('--oracle_local_steps', type=int, default=1000)
    parser.add_argument('--oracle_step_size', type=float, default=0.001)
    parser.add_argument('--homo_ratio', type=float, default=0.5)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--n_global_rounds', type=int, default=50)
    parser.add_argument('--use_ray', type=bool, default=True)

    args = parser.parse_args()

    writer = SummaryWriter(f'out/{args.dataset}_{args.step_size_0}_{args.worker_local_steps}_{algo}_{ts}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_size = tuple([int(a) for a in args.weak_learner_hid_dims.split("-")])

    # Load/split training data
    dataset_handle = DATASETS[args.dataset]
    dataset = dataset_handle(root='datasets/' + args.dataset, download=True)
    dataset_test = dataset_handle(root='datasets/' + args.dataset, train=False, download=True)
    if args.dataset == "mnist":
        data, label = dataset.train_data.to(dtype=torch.float32, device=device) / 255.0, \
                      dataset.train_labels.to(device=device)
        data = (data - torch.tensor(0.1307, device=device)) / torch.tensor(0.3081, device=device)
        assert (data.shape[0] == label.shape[0])
        (n_data, height, width) = data.shape
        n_class = 10
        n_channel = 1
        rand_index = torch.randperm(data.shape[0])
        data, label = data[rand_index], label[rand_index]
        get_init_weak_learner = lambda: utils.get_init_weak_learner(height, width, n_channel, n_class,
                                                                    hidden_size=hidden_size, type="MLP", device=device)

        data_test, label_test = dataset_test.train_data.to(dtype=torch.float32, device=device) / 255.0, \
                      dataset_test.train_labels.to(device=device)
        data_test = (data_test - torch.tensor(0.1307, device=device)) / torch.tensor(0.3081, device=device)
        assert (data_test.shape[0] == label_test.shape[0])

    elif args.dataset == "cifar":
        # processing training data
        data, label = torch.tensor(dataset.data, dtype=torch.float32, device=device) / 255.0, \
                      torch.tensor(dataset.targets, device=device)
        # normalize
        data = (data - torch.tensor((0.5, 0.5, 0.5), device=device)) / torch.tensor((0.5, 0.5, 0.5), device=device)
        data = data.permute(0, 3, 1, 2)  # from (H, W, C) to (C, H, W)

        # processing testing data
        data_test, label_test = torch.tensor(dataset_test.data, dtype=torch.float32, device=device) / 255.0, \
                      torch.tensor(dataset_test.targets, device=device)
        # normalize
        data_test = (data_test - torch.tensor((0.5, 0.5, 0.5), device=device)) / torch.tensor((0.5, 0.5, 0.5), device=device)
        data_test = data_test.permute(0, 3, 1, 2)  # from (H, W, C) to (C, H, W)

        assert (data.shape[0] == label.shape[0])
        (n_data, n_channel, height, width) = data.shape
        n_class = 10
        rand_index = torch.randperm(data.shape[0])
        data, label = data[rand_index], label[rand_index]
        get_init_weak_learner = lambda: utils.get_init_weak_learner(height, width, n_channel, n_class,
                                                                    hidden_size=hidden_size, type="Conv", device=device)
    else:
        raise NotImplementedError

    # data_list, label_list = data.chunk(args.n_workers), label.chunk(args.n_workers)
    data_list, label_list = utils.data_partition(data, label, args.n_workers, args.homo_ratio)


    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]

    Worker = utils_ray.Worker if args.use_ray else utils.Worker
    Server = utils_ray.Server if args.use_ray else utils.Server

    workers = [Worker(data_i, label_i, Dx_loss, get_init_weak_learner, args.worker_local_steps, args.oracle_local_steps,
                      args.oracle_step_size, device=device)
               for (data_i, label_i) in zip(data_list, label_list)]

    server = Server(workers, get_init_weak_learner, args.step_size_0, args.worker_local_steps, device=device)

    f_data = None
    f_data_test = None
    for round in tqdm(range(args.n_global_rounds)):
        server.global_step()
        # after every round, evaluate the current ensemble
        with torch.autograd.no_grad():
            f_data = server.f(data) if f_data is None else f_data + server.f_new(data)
            loss_round = loss(f_data, label)
            writer.add_scalar(f"global loss vs round @ {args.n_workers} workers/train", loss_round, round)
            pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = np.true_divide(pred.eq(label.view_as(pred)).sum().item(), label.shape[0])
            writer.add_scalar(f"correct rate vs round @ {args.n_workers} workers/train", correct, round)

            f_data_test = server.f(data_test) if f_data_test is None else f_data_test + server.f_new(data_test)
            loss_round = loss(f_data_test, label_test)
            writer.add_scalar(f"global loss vs round @ {args.n_workers} workers/test", loss_round, round)
            pred = f_data_test.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = np.true_divide(pred.eq(label_test.view_as(pred)).sum().item(), label_test.shape[0])
            writer.add_scalar(f"correct rate vs round @ {args.n_workers} workers/test", correct, round)

