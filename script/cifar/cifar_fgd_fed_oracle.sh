# This script records the recommended hyperparameters for "fgd_fed_oracle"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .1, no data augmentation
python run_fgd_fed_oracle.py --dataset cifar --homo_ratio .1 --n_workers 50 \
      --dense_hid_dims 120-84 --conv_hid_dims 6-16 --model convnet \
       --device_id 1,2,3,4,5 --num_oracle_steps 400 --local_opt_lr 1e-1 --p .5 --step_size_0 10



python run_fgd_fed_oracle.py --dataset cifar --homo_ratio .1 --n_workers 50 \
      --dense_hid_dims 120-84 --conv_hid_dims 6-16 --model convnet \
       --device_id 1,2,3,4,5 --num_oracle_steps 1000 --local_opt_lr 1e-1 --p .5 --step_size_0 10