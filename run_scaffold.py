import argparse
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from Dx_losses import Dx_cross_entropy
from tqdm import tqdm

from utils import load_data, data_partition, make_adv_label
from core import scaffold_ray, scaffold
import numpy as np
import time
import math

Dx_losses = {
    "logistic_regression": 123,
    "cross_entropy": Dx_cross_entropy
}
losses = {
    "logistic_regression": 123,
    "cross_entropy": lambda x, y: torch.nn.functional.cross_entropy(x, y)
}
DATASETS = {
    "cifar": datasets.CIFAR10,
    "mnist": datasets.MNIST
}

if __name__ == '__main__':
    ts = time.time()
    algo = "scaffold"
    parser = argparse.ArgumentParser(algo)

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--weak_learner_hid_dims', type=str, default='32-32')
    parser.add_argument('--step_size_0', type=float, default=0.0005)
    parser.add_argument('--loss', type=str, choices=['logistic_regression', 'l2_regression', 'cross_entropy'],
                        default='cross_entropy')
    parser.add_argument('--local_epoch', type=int, default=10)
    parser.add_argument('--homo_ratio', type=float, default=0.1)
    parser.add_argument('--n_workers', type=int, default=56)
    parser.add_argument('--step_per_epoch', type=int, default=1)
    parser.add_argument('--n_ray_workers', type=int, default=56)
    parser.add_argument('--n_global_rounds', type=int, default=5000)
    parser.add_argument('--use_ray', type=bool, default=False)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--comm_max', type=int, default=5000)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--use_adv_label', type=bool, default=False)

    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        args.use_ray = False
    else:
        device = torch.device("cpu")
    hidden_size = tuple([int(a) for a in args.weak_learner_hid_dims.split("-")])
    # Load/split data
    dataset_handle = DATASETS[args.dataset]
    dataset = dataset_handle(root='datasets/' + args.dataset, download=True)
    dataset_test = dataset_handle(root='datasets/' + args.dataset, train=False, download=True)

    data, label, data_test, label_test, n_class, get_init_weak_learner = load_data(args, hidden_size, device)

    data_list, label_list = data_partition(data, label, args.n_workers, args.homo_ratio)

    if args.use_adv_label:
        label_list = make_adv_label(label_list, n_class)

    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]

    init_model = get_init_weak_learner()

    args.worker_local_steps = args.local_epoch * args.step_per_epoch

    tb_file = f'out/{args.dataset}/s{args.homo_ratio}_adv{args.use_adv_label}/{args.weak_learner_hid_dims}/' \
              f'rhog{args.step_size_0}_K{args.worker_local_steps}_{algo}_{ts} '
    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)

    if args.use_ray:
        Worker = scaffold_ray.Worker
        Server = scaffold_ray.Server
        workers = [Worker(data=data_i, label=label_i, loss=loss, n_class=n_class, local_steps=args.worker_local_steps,
                          mb_size=int(data_i.shape[0] / args.step_per_epoch), device=device)
                   for (data_i, label_i) in zip(data_list, label_list)]
        server = Server(workers, init_model, args.step_size_0, args.worker_local_steps, device=device, p=args.p,
                        n_ray_workers=args.n_ray_workers)
    else:
        Worker = scaffold.Worker
        Server = scaffold.Server
        workers = [Worker(data=data_i, label=label_i, loss=loss, n_class=n_class, local_steps=args.worker_local_steps,
                          mb_size=int(data_i.shape[0] / args.step_per_epoch), device=device)
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
