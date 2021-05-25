# This script records the recommended hyperparameters for "fedprox"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .1, no data augmentation
python run_fedprox.py --dataset cifar --homo_ratio .1 --n_workers 50 --n_global_rounds 1000  --device cuda \
      --dense_hid_dims 120-84 --conv_hid_dims 6-16 --model convnet \
      --step_size_0 2e-2 --step_per_epoch 5 --local_epoch 5 --mu 2.

# This script records the recommended hyperparameters for "fedprox"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .1, no data augmentation
python run_fedprox.py --dataset cifar --homo_ratio .3 --n_workers 50 --n_global_rounds 1000  --device cuda \
      --dense_hid_dims 120-84 --conv_hid_dims 6-16 --model convnet \
      --step_size_0 1e-2 --step_per_epoch 5 --local_epoch 5 --mu 3.


# This script records the recommended hyperparameters for "fedprox"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .1, with data augmentation
python run_fedprox.py --dataset cifar --homo_ratio .1 --n_workers 50 --n_global_rounds 1000  --device cuda \
      --dense_hid_dims 120-84 --conv_hid_dims 6-16 --model convnet \
      --step_size_0 3e-2 --step_per_epoch 5 --local_epoch 5 --mu 2. --augment_data