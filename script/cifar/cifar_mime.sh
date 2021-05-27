# This script records the recommended hyperparameters for "mime"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .1
python run_mime.py --dataset cifar --homo_ratio .1 --n_workers 50 --n_global_rounds 500  --device cuda \
      --dense_hid_dims 120-84 --conv_hid_dims 6-16 --model convnet \
      --step_size_0 2e-2 --step_per_epoch 5 --local_epoch 5

# This script records the recommended hyperparameters for "mime"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .1, no data augmentation, load ckpt from ffgb_distill
python run_mime.py --dataset cifar --homo_ratio .1 --n_workers 50 --n_global_rounds 500  --device cuda \
      --dense_hid_dims 120-84 --conv_hid_dims 6-16 --model convnet \
      --step_size_0 2e-2 --step_per_epoch 5 --local_epoch 5 --load_ckpt_round 0

# This script records the recommended hyperparameters for "mime"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .3
python run_mime.py --dataset cifar --homo_ratio .3 --n_workers 50 --n_global_rounds 500  --device cuda \
      --dense_hid_dims 120-84 --conv_hid_dims 6-16 --model convnet \
      --step_size_0 2e-2 --step_per_epoch 5 --local_epoch 5