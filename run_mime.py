import argparse
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from Dx_losses import Dx_cross_entropy
from tqdm import tqdm

from utils import load_data, data_partition, make_adv_label, is_nan
from core import mime, mime_ray
import numpy as np
import time
import copy
import os
from model import convnet
from model import mlp
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"


if __name__ == '__main__':
    ts = time.time()
    algo = "mime"
    parser = argparse.ArgumentParser(algo)

    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dense_hid_dims', type=str, default='120-84')
    parser.add_argument('--conv_hid_dims', type=str, default='6-16')
    parser.add_argument('--model', type=str, choices=['mlp', 'convnet'], default='convnet')
    parser.add_argument('--step_size_0', type=float, default=5e-2)
    parser.add_argument('--loss', type=str, choices=['logistic_regression', 'l2_regression', 'cross_entropy'],
                        default='cross_entropy')
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--homo_ratio', type=float, default=0.1)
    parser.add_argument('--n_workers', type=int, default=56)
    parser.add_argument('--step_per_epoch', type=int, default=5)
    parser.add_argument('--n_ray_workers', type=int, default=56)
    parser.add_argument('--n_global_rounds', type=int, default=500)
    parser.add_argument('--use_ray', action="store_true")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--comm_max', type=int, default=2100)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--use_adv_label', type=bool, default=False)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--augment_data', action='store_true')
    parser.add_argument('--load_ckpt_round', type=int, default=-1)

    args = parser.parse_args()

    if "cuda" in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if args.load_ckpt_round >= 0:
        f_state_dict = torch.load(f"./ckpt/ffgb_distill_ckpt{args.load_ckpt_round}.pt", map_location=device)
    # torch.manual_seed(args.seed)

    dense_hidden_size = tuple([int(a) for a in args.dense_hid_dims.split("-")])
    conv_hidden_size = tuple([int(a) for a in args.conv_hid_dims.split("-")])

    data, label, data_test, label_test, n_class, get_init_weak_learner = load_data(args, dense_hidden_size,
                                                                                   device,
                                                                                   augment_data=args.augment_data)
    if args.model == "convnet":
        get_init_weak_learner = partial(convnet.LeNet5, n_class, data.shape[1], conv_hidden_size, dense_hidden_size, device)
    elif args.model == "mlp":
        get_init_weak_learner = partial(mlp.MLP, n_class, dense_hidden_size, device)

    data_list, label_list = data_partition(data, label, args.n_workers, args.homo_ratio)

    if args.use_adv_label:
        label_list = make_adv_label(label_list, n_class)

    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]

    init_model = get_init_weak_learner()

    args.worker_local_steps = args.local_epoch * args.step_per_epoch

    if args.use_ray:
        Worker = mime_ray.Worker
        Server = mime_ray.Server
        workers = [
            Worker(data_i, label_i, loss, args.worker_local_steps, mb_size=int(data_i.shape[0] / args.step_per_epoch),
                   device=device,
                   beta=args.beta)
            for (data_i, label_i) in zip(data_list, label_list)]
        server = Server(workers, init_model, args.step_size_0, args.worker_local_steps, device=device, p=args.p,
                        beta=args.beta, n_ray_workers=args.n_ray_workers)
    else:
        Worker = mime.Worker
        Server = mime.Server
        workers = [
            Worker(data_i, label_i, loss, args.worker_local_steps, mb_size=int(data_i.shape[0] / args.step_per_epoch),
                   device=device,
                   beta=args.beta)
            for (data_i, label_i) in zip(data_list, label_list)]
        server = Server(workers, init_model, args.step_size_0, args.worker_local_steps, device=device, p=args.p,
                        beta=args.beta)

    comm_cost = 5
    if args.load_ckpt_round >= 0:
        server.f.load_state_dict(f_state_dict)
        round_0 = (args.load_ckpt_round + 1) * 5
    else:
        round_0 = 0

    tb_file = f'out/{args.dataset}/{args.conv_hid_dims}_{args.dense_hid_dims}/s{args.homo_ratio}' \
                  f'/N{args.n_workers}/rhog{args.step_size_0}_{algo}_{ts}'

    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    for round in tqdm(range(args.n_global_rounds)):
        # f_param_prev = copy.deepcopy(server.f.state_dict())

        server.global_step()
        with torch.autograd.no_grad():
            if round % args.eval_freq == 0:
                # f_data = server.f(data)
                # loss_round = loss(f_data, label)
                # if is_nan(loss_round):
                #     states = [f_param_prev, round, comm_cost, tb_file, args.seed]
                #     if not args.load_ckpt: torch.save(states, f'ckpt_{algo}.pt')
                #     raise RuntimeError
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
        comm_cost += 4

    print(args)
