import argparse
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from Dx_losses import Dx_cross_entropy
from tqdm import tqdm

from utils import load_data, data_partition, make_adv_label
from core import fed_avg, fed_avg_ray
import numpy as np
import time
from model import convnet
from functools import partial

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
    algo = "fed-avg"
    parser = argparse.ArgumentParser(algo)
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dense_hid_dims', type=str, default='120-84')
    parser.add_argument('--conv_hid_dims', type=str, default='6-16')
    parser.add_argument('--model', type=str, default='convnet')
    parser.add_argument('--step_size_0', type=float, default=0.0005)
    parser.add_argument('--loss', type=str, choices=['logistic_regression', 'l2_regression', 'cross_entropy'],
                        default='cross_entropy')
    parser.add_argument('--homo_ratio', type=float, default=0.1)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--n_workers', type=int, default=50)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--step_per_epoch', type=int, default=5)
    parser.add_argument('--n_ray_workers', type=int, default=2)
    parser.add_argument('--n_global_rounds', type=int, default=5000)
    parser.add_argument('--use_ray', type=bool, default=False)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--comm_max', type=int, default=5000)
    parser.add_argument('--use_adv_label', type=bool, default=False)
    parser.add_argument('--augment_data', action='store_true')

    args = parser.parse_args()

    args.worker_local_steps = args.local_epoch * args.step_per_epoch


    # writer = SummaryWriter(
    #     f'out/{args.dataset}/s{args.homo_ratio}_adv{args.use_adv_label}/{args.weak_learner_hid_dims}/rhog{args.step_size_0}_K{args.worker_local_steps}_{algo}_{ts}'
    # )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dense_hidden_size = tuple([int(a) for a in args.dense_hid_dims.split("-")])
    conv_hidden_size = tuple([int(a) for a in args.conv_hid_dims.split("-")])
    # Load/split data

    data, label, data_test, label_test, n_class, get_init_weak_learner = load_data(args, dense_hidden_size,
                                                                                   device, augment_data=args.augment_data)

    if args.model == "convnet":
        get_init_weak_learner = partial(convnet.LeNet5, n_class, data.shape[1], conv_hidden_size, dense_hidden_size, device)

    data_list, label_list = data_partition(data, label, args.n_workers, args.homo_ratio)

    if args.use_adv_label:
        label_list = make_adv_label(label_list, n_class)

    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]

    Worker = fed_avg_ray.Worker if args.use_ray else fed_avg.Worker
    Server = fed_avg_ray.Server if args.use_ray else fed_avg.Server

    init_model = get_init_weak_learner()



    workers = [Worker(data=data_i, label=label_i, loss=loss, n_class=n_class, local_steps=args.worker_local_steps,
                      mb_size=int(data_i.shape[0]/args.step_per_epoch), device=device)
               for (data_i, label_i) in zip(data_list, label_list)]
    if args.use_ray:
        server = Server(workers, init_model, args.step_size_0, args.worker_local_steps,
                        device=device, n_ray_workers=args.n_ray_workers
                        )
    else:
        server = Server(workers, init_model, args.step_size_0, args.worker_local_steps, device=device, p=args.p)
    comm_cost = 2

    tb_file = f'out/{args.dataset}/{args.conv_hid_dims}_{args.dense_hid_dims}/s{args.homo_ratio}' \
              f'/N{args.n_workers}/rhog{args.step_size_0}_{algo}_{ts}'

    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    print(args)
    for round in tqdm(range(args.n_global_rounds)):
        server.global_step()
        with torch.autograd.no_grad():

            if round % args.eval_freq == 0:
                # f_data = server.f(data)
                # loss_round = loss(f_data, label)
                # writer.add_scalar(
                #     f"global loss vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                #     loss_round, round)
                # writer.add_scalar(
                #     f"global loss vs comm, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                #     loss_round, comm_cost)
                # pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # correct = np.true_divide(pred.eq(label.view_as(pred)).sum().item(), label.shape[0])
                # writer.add_scalar(
                #     f"correct rate vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                #     correct, round)
                # writer.add_scalar(
                #     f"correct rate vs comm, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                #     correct, comm_cost)

                # if f_data_test is None, server.f is a constant zero function
                f_data_test = server.f(data_test)
                loss_round = loss(f_data_test, label_test)
                writer.add_scalar(
                    f"global loss vs round/test",
                    loss_round, round)
                writer.add_scalar(
                    f"global loss vs comm/test",
                    loss_round, comm_cost)
                pred = f_data_test.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = np.true_divide(pred.eq(label_test.view_as(pred)).sum().item(), label_test.shape[0])
                writer.add_scalar(
                    f"correct rate vs round/test",
                    correct, round)
                writer.add_scalar(
                    f"correct rate vs comm/test",
                    correct, comm_cost)
                # print("Round %5d, accuracy %.3f" % (round, correct))

            if comm_cost > args.comm_max:
                break
            comm_cost += 2
    print(args)
