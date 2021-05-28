# This script records the recommended hyperparameters for "ffgb_distill"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .1, no data augmentation
python run_ffgd_distill.py --dataset mnist --homo_ratio .1 --n_workers 50 --n_global_rounds 100  --device cuda \
      --dense_hid_dims 32-32 --model mlp \
      --n_ray_workers 8 --worker_local_steps 5 --step_size_0 10 --p .9 --oracle_local_steps 2000\
       --device_id 1,2,3,4,5 --backend ray --oracle_step_size 2e-3


# This script records the recommended hyperparameters for "ffgb_distill"
# Model: LeNet5 with conv_hid_dim (6, 16) and dense_hid_dim (120, 84)
# Experiment setting: N = 50, s = .3, no data augmentation
python run_ffgd_distill.py --dataset mnist --homo_ratio .3 --n_workers 50 --n_global_rounds 100  --device cuda \
      --dense_hid_dims 32-32 --model mlp \
      --n_ray_workers 5 --worker_local_steps 5 --step_size_0 10 --p .9 --oracle_local_steps 2000\
       --device_id 1,2,3,4,5 --backend ray --oracle_step_size 2e-3