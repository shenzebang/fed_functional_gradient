import argparse
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from Dx_losses import Dx_cross_entropy
from tqdm import tqdm

import utils
from core import mime
import numpy as np
import time
import math

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
    algo = "mime"
    parser = argparse.ArgumentParser(algo)

    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--weak_learner_hid_dims', type=str, default='32-32')
    parser.add_argument('--step_size_0', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, choices=['logistic_regression', 'l2_regression', 'cross_entropy'],
                        default='cross_entropy')
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--homo_ratio', type=float, default=0.5)
    parser.add_argument('--n_workers', type=int, default=56)
    parser.add_argument('--local_mb_size', type=int, default=-1, help='negative mb size means full gradient descent locally')
    parser.add_argument('--n_ray_workers', type=int, default=2)
    parser.add_argument('--n_global_rounds', type=int, default=5000)
    parser.add_argument('--use_ray', type=bool, default=False)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--comm_max', type=int, default=5000)
    parser.add_argument('--p', type=float, default=0.1)


    args = parser.parse_args()



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_size = tuple([int(a) for a in args.weak_learner_hid_dims.split("-")])
    # Load/split data
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
        init_model = utils.get_init_weak_learner(height, width, n_channel, n_class, hidden_size=hidden_size,
                                                 type="MLP", device=device)


        data_test, label_test = dataset_test.train_data.to(dtype=torch.float32, device=device) / 255.0, \
                      dataset_test.train_labels.to(device=device)
        data_test = (data_test - torch.tensor(0.1307, device=device)) / torch.tensor(0.3081, device=device)
        assert (data_test.shape[0] == label_test.shape[0])

    elif args.dataset == "cifar":
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
        init_model = utils.get_init_weak_learner(height, width, n_channel, n_class,
                                                 hidden_size=hidden_size, type="Conv", device=device)
    else:
        raise NotImplementedError

    data_list, label_list = utils.data_partition(data, label, args.n_workers, args.homo_ratio)

    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]

    Worker = mime.Worker
    Server = mime.Server

    data_per_machine = int(n_data/args.n_workers)
    if args.local_mb_size > 0:
        args.worker_local_steps = math.ceil(data_per_machine*args.local_epoch/args.local_mb_size)
    else:  # full gradient
        args.worker_local_steps = args.local_epoch

    print(f"n local steps is {args.worker_local_steps}")

    writer = SummaryWriter(
        f'out/rhog{args.step_size_0}_K{args.worker_local_steps}_mb{args.local_mb_size}_{algo}_{ts}'
    )

    workers = [Worker(data_i, label_i, loss, args.worker_local_steps, mb_size=args.local_mb_size, device=device)
               for (data_i, label_i) in zip(data_list, label_list)]
    server = Server(workers, init_model, args.step_size_0, args.worker_local_steps, device=device, p=args.p)
    comm_cost = 5
    for round in tqdm(range(args.n_global_rounds)):
        server.global_step()
        with torch.autograd.no_grad():
            if round % args.eval_freq == 0:
                f_data = server.f(data)
                loss_round = loss(f_data, label)
                writer.add_scalar(
                    f"global loss vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                    loss_round, round)
                writer.add_scalar(
                    f"global loss vs comm, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                    loss_round, comm_cost)
                pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = np.true_divide(pred.eq(label.view_as(pred)).sum().item(), label.shape[0])
                writer.add_scalar(
                    f"correct rate vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                    correct, round)
                writer.add_scalar(
                    f"correct rate vs comm, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                    correct, comm_cost)

                # if f_data_test is None, server.f is a constant zero function
                f_data_test = server.f(data_test)
                loss_round = loss(f_data_test, label_test)
                writer.add_scalar(
                    f"global loss vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/test",
                    loss_round, round)
                writer.add_scalar(
                    f"global loss vs comm, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/test",
                    loss_round, comm_cost)
                pred = f_data_test.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = np.true_divide(pred.eq(label_test.view_as(pred)).sum().item(), label_test.shape[0])
                writer.add_scalar(
                    f"correct rate vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/test",
                    correct, round)
                writer.add_scalar(
                    f"correct rate vs comm, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/test",
                    correct, comm_cost)
        if comm_cost > args.comm_max:
            break
        comm_cost += 4

    print(args)