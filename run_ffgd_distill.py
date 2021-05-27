import argparse
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from Dx_losses import Dx_cross_entropy
from tqdm import tqdm

from utils import load_data, data_partition, make_adv_label, make_dataloaders
from torchvision import transforms
from core import ffgd_distill_ray
# from resnet import resnet20

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import time

import os
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



if __name__ == '__main__':
    ts = time.time()
    algo = 'ffgb_distill'
    parser = argparse.ArgumentParser(algo)

    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dense_hid_dims', type=str, default='120-84')
    parser.add_argument('--conv_hid_dims', type=str, default='6-16')
    parser.add_argument('--model', type=str, default='convnet')
    parser.add_argument('--step_size_0', type=float, default=20.0)

    parser.add_argument('--loss', type=str, choices=['logistic_regression', 'l2_regression', 'cross_entropy'],
                        default='cross_entropy')
    parser.add_argument('--worker_local_steps', type=int, default=10)
    parser.add_argument('--oracle_local_steps', type=int, default=1000)
    parser.add_argument('--oracle_step_size', type=float, default=0.001)
    parser.add_argument('--homo_ratio', type=float, default=0.1)
    parser.add_argument('--p', type=float, default=1, help='step size decay exponential')
    parser.add_argument('--n_workers', type=int, default=56)
    parser.add_argument('--oracle_mb_size', type=int, default=128)
    parser.add_argument('--n_ray_workers', type=int, default=56)
    parser.add_argument('--n_global_rounds', type=int, default=100)
    # parser.add_argument('--use_ray', type=bool, default=True)
    parser.add_argument('--backend', type=str, default="None")
    parser.add_argument('--store_f', type=bool, default=False, help="store the variable function. high memory cost.")
    parser.add_argument('--comm_max', type=int, default=0, help="0 means no constraint on comm cost")
    parser.add_argument('--device_ids', type=str, default="-1")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--use_adv_label', type=bool, default=False)
    parser.add_argument('--augment_data', action='store_true')

    args = parser.parse_args()

    device_ids = [int(a) for a in args.device_ids.split(",")]
    if device_ids[0] != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_ids}"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    dense_hidden_size = tuple([int(a) for a in args.dense_hid_dims.split("-")])
    conv_hidden_size = tuple([int(a) for a in args.conv_hid_dims.split("-")])

    # Load/split training data

    data, label, data_test, label_test, n_class, get_init_weak_learner = load_data(args, dense_hidden_size,
                                                                                   device,
                                                                                   augment_data=args.augment_data)

    distill_ratio = .2
    data, x_distill = torch.split(data, [int(data.shape[0] * (1-distill_ratio)), int(data.shape[0] * distill_ratio)])
    label = label[0: int(label.shape[0] * (1-distill_ratio))]

    # print(data.shape, label.shape)

    if args.model == "convnet":
        get_init_weak_learner = partial(convnet.LeNet5, n_class, data.shape[1], conv_hidden_size, dense_hidden_size, device)


    # data_list, label_list = data.chunk(args.n_workers), label.chunk(args.n_workers)
    # data_list, label_list = data_partition(data, label, args.n_workers, args.homo_ratio)
    data_list, label_list = data_partition(data, label, args.n_workers, args.homo_ratio)
    # get_init_weak_learner = lambda: resnet20().to(device)
    # dataloaders = make_dataloaders(data_list, label_list, data_transforms)
    if args.use_adv_label:
        label_list = make_adv_label(label_list, n_class)

    Dx_loss = Dx_losses[args.loss]
    loss = losses[args.loss]

    if args.backend == "ray":
        args.use_ray = True
        args.use_joblib = False
        Worker = ffgd_distill_ray.Worker
        Server = ffgd_distill_ray.Server
    else:
        raise NotImplementedError

    workers = [Worker(data=data_i, label=label_i, Dx_loss=Dx_loss, get_init_weak_learner=get_init_weak_learner,
                      n_class=n_class, local_steps=args.worker_local_steps, oracle_steps=args.oracle_local_steps,
                      oracle_step_size=args.oracle_step_size, device=device, mb_size=args.oracle_mb_size)
               for (data_i, label_i) in zip(data_list, label_list)]

    server = Server(workers, get_init_weak_learner, x_distill, args.step_size_0, args.worker_local_steps, n_ray_workers=args.n_ray_workers,
                        device=device, step_size_decay_p=args.p)

    comm_cost = 2

    tb_file = f'out/{args.dataset}/{args.conv_hid_dims}_{args.dense_hid_dims}/s{args.homo_ratio}' \
              f'/N{args.n_workers}/rhog{args.step_size_0}_{algo}_{ts}'

    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    print(args)
    for round in tqdm(range(args.n_global_rounds)):
        server.global_step()
        # after every round, evaluate the current ensemble
        if round % args.eval_freq == 0:
            with torch.autograd.no_grad():
                # if f_data is None, server.f is a constant zero function
                # f_new_eval = server.f_new
                # f_data = f_new_eval(data_eval) if f_data is None else f_data + f_new_eval(data_eval)
                # loss_round = loss(f_data, label_eval)
                # writer.add_scalar(
                #     f"global loss vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                #     loss_round, round)
                # writer.add_scalar(
                #     f"global loss vs comm, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                #     loss_round, comm_cost)
                # pred = f_data.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # correct = np.true_divide(pred.eq(label_eval.view_as(pred)).sum().item(), label_eval.shape[0])
                # writer.add_scalar(
                #     f"correct rate vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                #     correct, round)
                # writer.add_scalar(
                #     f"correct rate vs comm, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}/train",
                #     correct, comm_cost)
                #
                # writer.add_scalar(
                #     f"residual vs round, {args.dataset}, N={args.n_workers}, s={args.homo_ratio}", residual, round)
                # if f_data_test is None, server.f is a constant zero function
                # if f_data_test is None:
                #     f_data_test = server.f_new(data_test)
                # else:
                #     f_data_test = f_data_test + server.f_new(data_test)
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
                print(correct)
                writer.add_scalar(
                    f"correct rate vs round/test",
                    correct, round)
                writer.add_scalar(
                    f"correct rate vs comm/test",
                    correct, comm_cost)
            torch.save(server.f.function_list[0].state_dict(), f'./ckpt/ffgb_distill_ckpt{round}.pt')

        if comm_cost > args.comm_max > 0:
            break

        comm_cost += args.n_workers * args.worker_local_steps

    print(args)

