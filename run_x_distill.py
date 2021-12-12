import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from Dx_losses import Dx_cross_entropy
from config import make_parser
from utils.data_utils import load_dataset, make_transforms, make_dataloader, split_dataset
from utils.model_utils import make_model as _make_model
from core.ffgb_distill import FFGB_D
from core.fed_avg_distill import FEDAVG_D
from utils.logger_utils import make_evaluate_fn, make_monitor_fn, Logger
import json
import time
import os



LEARNERS = {
    'ffgb_d': FFGB_D,
    'fedavg_d': FEDAVG_D
}


if __name__ == '__main__':
    args = make_parser()
    learner = LEARNERS[args.learner]
    print("#" * 30)
    print("Run FFGB-D")

    device_ids = [int(a) for a in args.device_ids.split(",")]
    if device_ids[0] != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_ids}"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    Dx_loss = Dx_cross_entropy
    loss = torch.nn.functional.cross_entropy

    # 1. set saving directory
    print("#"*30)
    print("making saving directory")
    level = args.homo_ratio if args.heterogeneity == "mix" else args.dir_level
    if args.learner == "fedavg_d":
        algo_config = f"_{args.fedavg_d_local_lr}_{args.fedavg_d_local_epoch}_{args.fedavg_d_weight_decay}"
    elif args.learner == "ffgb_d":
        algo_config = f"_{args.local_steps}_{args.functional_lr}_{args.f_l2_reg}_{args.weak_learner_epoch}_{args.weak_learner_lr}_{args.weak_learner_weight_decay}"
    else:
        raise NotImplementedError

    experiment_setup = f"FFL_{args.heterogeneity}_{level}_{args.n_workers}_{args.n_workers_per_round}_{args.dataset}_{args.model}"
    hyperparameter_setup = f"{args.learner}_{args.local_dataloader_batch_size}_{args.distill_dataloader_batch_size}" + algo_config

    args.save_dir = 'output/%s/%s' % (experiment_setup, hyperparameter_setup)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.save_dir + '/config.json', 'w') as f:
        json.dump(vars(args), f)

    tb_file = args.save_dir + f'/{time.time()}'
    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)


    # 2. create dataloaders
    print("#" * 30)
    print("making dataloders")
    dataset_trn, dataset_tst, n_classes, n_channels, img_size = load_dataset(args.dataset)
    dataset_distill, _, _, _, _ = load_dataset(args.dataset_distill)

    transforms = make_transforms(args.dataset, train=True)  # transforms for data augmentation and normalization
    local_datasets = split_dataset(args.n_workers, args.homo_ratio, dataset_trn, transforms)
    client_dataloaders = [make_dataloader(args, "train", local_dataset) for local_dataset in local_datasets]

    transforms_test = make_transforms(args.dataset, train=False)
    dataset_tst.transform = transforms_test
    test_dataloader = make_dataloader(args, "test", dataset_tst)

    transforms_distill = make_transforms(args.dataset_distill, train=True, is_distill=True)
    dataset_distill.transform = transforms_distill
    distill_dataloader = make_dataloader(args, "distill", dataset_distill)


    # 3. create loggers
    test_fn_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='accuracy', n_classes=n_classes, loss_fn=loss)
    statistics_monitor_fn = make_monitor_fn()

    logger_accuracy = Logger(writer, test_fn_accuracy, test_metric='accuracy')
    logger_monitor = Logger(writer, statistics_monitor_fn, test_metric='model_monitor')
    loggers = [logger_accuracy, logger_monitor]

    # 4. create model and trainer
    print("#" * 30)
    print("creating model and trainer")
    make_model = lambda: _make_model(args, n_classes, n_channels, img_size, device)
    model_init = make_model()

    ffgb_d = learner(model_init, make_model, client_dataloaders, distill_dataloader, Dx_loss, loggers, args, device)

    # 5. train
    print("#" * 30)
    print("start training")
    ffgb_d.fit()
    print("done training")

    # 6. save model
    if args.save_model:
        model_file = f"./model_{args.dataset}.pth"
        torch.save(ffgb_d.server_state.model.state_dict(), model_file)