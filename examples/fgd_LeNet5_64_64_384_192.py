import argparse
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from Dx_losses import Dx_cross_entropy
from tqdm import tqdm

from utils import load_data, data_partition, make_adv_label
from core import rfgd
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
    algo = 'rfgd'
    parser = argparse.ArgumentParser(algo)

    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dense_hid_dims', type=str, default='384-192')
    parser.add_argument('--conv_hid_dims', type=str, default='64-64')
    parser.add_argument('--model', type=str, default='convnet')

    parser.add_argument('--step_size_0', type=float, default=10.0)
    parser.add_argument('--loss', type=str, choices=['logistic_regression', 'l2_regression', 'cross_entropy'],
                        default='cross_entropy')
    parser.add_argument('--oracle_local_steps', type=int, default=30000)
    parser.add_argument('--oracle_step_size', type=float, default=1e-3)
    parser.add_argument('--p', type=float, default=.5, help='step size decay exponential')
    parser.add_argument('--oracle_mb_size', type=int, default=128)
    parser.add_argument('--n_global_rounds', type=int, default=100)
    parser.add_argument('--store_f', type=bool, default=False, help="store the variable function. high memory cost.")
    parser.add_argument('--comm_max', type=int, default=0, help="0 means no constraint on comm cost")
    parser.add_argument('--eval_freq', type=int, default=1)

    args = parser.parse_args()
    tb_file = f'out/{args.dataset}/{args.conv_hid_dims}_{args.dense_hid_dims}' \
              f'/rhog{args.step_size_0}_{algo}_{ts}'
    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dense_hidden_size = tuple([int(a) for a in args.dense_hid_dims.split("-")])
    conv_hidden_size = tuple([int(a) for a in args.conv_hid_dims.split("-")])
    # Load/split training data

    data, label, data_test, label_test, n_class, get_init_weak_learner = load_data(args, dense_hidden_size, device)
    # data_list, label_list = data.chunk(args.n_workers), label.chunk(args.n_workers)
    if args.model == "convnet":
        get_init_weak_learner = partial(convnet.LeNet5, n_class, data.shape[1], conv_hidden_size, dense_hidden_size, device)


    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]

    rfgd = rfgd.RFGD(data=data, label=label, Dx_loss=Dx_loss, n_class=n_class, get_init_weak_learner=get_init_weak_learner,
                     oracle_steps=args.oracle_local_steps, oracle_step_size=args.oracle_step_size,
                     use_residual=True, step_size_0=args.step_size_0, device=device, mb_size=args.oracle_mb_size, p=args.p)

    f_data = None
    f_data_test = None
    comm_cost = 2


    for round in tqdm(range(args.n_global_rounds)):
        f_new = rfgd.step()
        # after every round, evaluate the current ensemble
        if round % args.eval_freq == 0:
            with torch.autograd.no_grad():
                # if f_data_test is None, server.f is a constant zero function
                f_data_test = f_new(data_test) if f_data_test is None else f_data_test + f_new(data_test)
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

    print(args)

