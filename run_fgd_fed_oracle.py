import argparse
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from Dx_losses import Dx_cross_entropy
from tqdm import tqdm

from utils import load_data, data_partition
from core import fgd_fed_oracle
from model import convnet
from functools import partial


import numpy as np
import time

Dx_losses = {
    "logistic_regression": 123,
    "cross_entropy": Dx_cross_entropy
}
losses = {
    "logistic_regression": 123,
    "cross_entropy": lambda x, y: torch.nn.functional.cross_entropy(x, y)
}


# todo: manage the gpu id
# todo: BUG when n_data mod n_workers is non-zero

if __name__ == '__main__':
    ts = time.time()
    algo = 'fgd_fed_oracle'
    parser = argparse.ArgumentParser(algo)

    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dense_hid_dims', type=str, default='120-84')
    parser.add_argument('--conv_hid_dims', type=str, default='6-16')
    parser.add_argument('--model', type=str, default='convnet')
    parser.add_argument('--homo_ratio', type=float, default=0.1)
    parser.add_argument('--step_size_0', type=float, default=20.0)
    parser.add_argument('--loss', type=str, choices=['logistic_regression', 'l2_regression', 'cross_entropy'],
                        default='cross_entropy')
    parser.add_argument('--num_oracle_steps', type=int, default=200)
    parser.add_argument('--num_local_epochs', type=int, default=5)
    parser.add_argument('--epoch_per_step', type=float, default=.2)
    parser.add_argument('--local_opt_lr', type=float, default=5e-2)
    parser.add_argument('--p', type=float, default=1, help='step size decay exponential')
    parser.add_argument('--n_global_rounds', type=int, default=100)
    parser.add_argument('--store_f', type=bool, default=False, help="store the variable function. high memory cost.")
    parser.add_argument('--comm_max', type=int, default=0, help="0 means no constraint on comm cost")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--use_adv_label', type=bool, default=False)
    parser.add_argument('--n_workers', type=int, default=50)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dense_hidden_size = tuple([int(a) for a in args.dense_hid_dims.split("-")])
    conv_hidden_size = tuple([int(a) for a in args.conv_hid_dims.split("-")])
    # Load/split training data

    data, label, data_test, label_test, n_class, get_init_weak_learner = load_data(args, dense_hidden_size, device)
    # data_list, label_list = data.chunk(args.n_workers), label.chunk(args.n_workers)
    if args.model == "convnet":
        get_init_weak_learner = partial(convnet.LeNet5, n_class, data.shape[1], conv_hidden_size, dense_hidden_size, device)

    data_list, label_list = data_partition(data, label, args.n_workers, args.homo_ratio)

    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]

    fed_oracle = fgd_fed_oracle.FedOracle(data_list, label_list, Dx_loss, args.num_local_epochs, args.epoch_per_step,
                                          args.local_opt_lr, args.num_oracle_steps, n_class)
    server = fgd_fed_oracle.Server(fed_oracle, get_init_weak_learner, args.step_size_0, args.p)

    f_data = None
    f_data_test = None
    comm_cost = 2

    tb_file = f'out/{args.dataset}/{args.conv_hid_dims}_{args.dense_hid_dims}/s{args.homo_ratio}' \
              f'/N{args.n_workers}/rhog{args.step_size_0}_{algo}_{ts}'

    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    print(args)
    for round in tqdm(range(args.n_global_rounds)):
        f_new = server.step()
        # after every round, evaluate the current ensemble
        if round % args.eval_freq == 0:
            with torch.autograd.no_grad():
                # if f_data is None, server.f is a constant zero function
                # f_data = f_new(data) if f_data is None else f_data + f_new(data)
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
                #
                # writer.add_scalar(
                #     f"residual vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}",
                #     residual.item(), round)

                # if f_data_test is None, server.f is a constant zero function
                f_data_test = f_new(data_test) if f_data_test is None else f_data_test + f_new(data_test)
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
            print("Round %5d, accuracy %.3f" % (round, correct))

        if comm_cost > args.comm_max and args.comm_max > 0:
            break

        comm_cost += args.num_oracle_steps * 4

    print(args)

