# This script records the recommended hyperparameters for "fedprox"
# Model: MLP with (32, 32) hidden neurons
# Experiment setting: N = 50, s = .1, no data augmentation
python run_fedprox.py --dataset mnist --homo_ratio .1 --n_workers 50 --n_global_rounds 200  --device cuda \
      --dense_hid_dims 32-32 --model mlp \
      --step_size_0 5e-2 --step_per_epoch 5 --local_epoch 5 --mu 2.

# This script records the recommended hyperparameters for "fedprox"
# Model: MLP with (32, 32) hidden neurons
# Experiment setting: N = 50, s = .1, no data augmentation
python run_fedprox.py --dataset mnist --homo_ratio .3 --n_workers 50 --n_global_rounds 200  --device cuda \
      --dense_hid_dims 32-32 --model mlp \
      --step_size_0 5e-2 --step_per_epoch 5 --local_epoch 5 --mu 3.