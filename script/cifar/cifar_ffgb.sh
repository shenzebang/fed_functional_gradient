# This script records the recommended hyperparameters for "ffgb"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .1
python run_ffgd.py --dataset cifar --homo_ratio .1 --n_workers 50 --n_global_rounds 100  --device cuda \
      --dense_hid_dims 120-84 --conv_hid_dims 6-16 --model convnet \
      --n_ray_workers 8 --worker_local_steps 4 --step_size_0 10 --p .9 --oracle_local_steps 6000\
       --device_id 1,2,3,4,5,6,7 --backend ray --oracle_step_size 2e-3
