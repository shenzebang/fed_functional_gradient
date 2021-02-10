# Federated Functional Gradient Boosting
Please first install the requirements.txt.

To run experiments, please run mnist_batch_ffgd.sh and mnist_batch_fed_avg.sh in the script folder.

The number of ray workers should be the number of CPUs available for the best efficiency of the code.
This is currently set to 4 in the script.

Important: Please ensure that mod(n_workers, n_ray_workers) == 0. 