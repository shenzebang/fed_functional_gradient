import argparse

def make_parser():
    parser = argparse.ArgumentParser()
    # general configurations
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_global_rounds', type=int, default=100)
    parser.add_argument('--device_ids', type=str, default="-1")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test_dataloader_batch_size', type=int, default=200)
    # tricks for NN training
    parser.add_argument('--no_data_augmentation', action='store_true', help='disable the data augmentation')

    # Experiment setup
    parser.add_argument('--learner', type=str, choices=['ffgb_d', 'fedavg_d'], default='ffgb_d')
    parser.add_argument('--heterogeneity', type=str, choices=['mix', 'dir'], default='mix',
                        help='Type of heterogeneity, mix or dir(dirichlet)')
    parser.add_argument('--dir_level', type=float, default=.3, help='hyperparameter of the Dirichlet distribution')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataset_distill', type=str, default='cifar100')
    parser.add_argument('--dense_hid_dims', type=str, default='120-84')
    parser.add_argument('--conv_hid_dims', type=str, default='6-16')
    parser.add_argument('--model', type=str, default='convnet')
    parser.add_argument('--homo_ratio', type=float, default=0.1)
    parser.add_argument('--n_workers', type=int, default=50)
    parser.add_argument('--n_workers_per_round', type=int, default=5)

    # General hyperparameters
    parser.add_argument('--local_dataloader_batch_size', type=int, default=64)
    parser.add_argument('--distill_dataloader_batch_size', type=int, default=64)

    # Hyperparameters for FFGB-D
    parser.add_argument('--functional_lr', type=float, default=10.)
    parser.add_argument('--local_steps', type=int, default=1)
    parser.add_argument('--weak_learner_epoch', type=int, default=30)
    parser.add_argument('--weak_learner_lr', type=float, default=.1)
    parser.add_argument('--weak_learner_weight_decay', type=float, default=.001)

    # Hyperparameters for FEDAVG-D

    parser.add_argument('--fedavg_d_local_lr', type=float, default=.01)
    parser.add_argument('--fedavg_d_local_epoch', type=int, default=50)
    parser.add_argument('--fedavg_d_weight_decay', type=float, default=1e-3)

    # distill oracle
    parser.add_argument('--distill_oracle', type=str, choices=["kl", "l2"], default="l2")
    parser.add_argument('--distill_oracle_epoch', type=int, default=10)
    parser.add_argument('--distill_oracle_lr', type=float, default=1e-3)
    parser.add_argument('--distill_oracle_weight_decay', type=float, default=.001)

    return parser.parse_args()