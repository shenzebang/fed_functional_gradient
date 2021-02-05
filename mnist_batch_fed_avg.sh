
#for eta in 25
#do
#for K in   2 3
#do
#for p in .75 .8 .85 .9 .95 1
#do
#    python run_ffgd.py --n_global_rounds 20000 --dataset mnist  --homo_ratio .1 --step_size_0 $eta\
#     --p $p --oracle_mb_size 128 --comm_max 1200 --worker_local_steps $K --n_workers 56 --n_ray_workers 56 \
#     --oracle_step_size 5e-3 --oracle_local_steps 2000 --device cpu --backend joblib
#done
#done
#done

for s in 0.1
do
python run_fed_avg.py --n_global_rounds 20000 --dataset mnist  --homo_ratio $s --step_size_0 8e-5 --p 0 \
    --local_epoch 5 --comm_max 3000 --step_per_epoch 5 --weak_learner_hid_dims 32-32 --use_adv_label True
done

#for i in 10 20 30 40
#do
#for j in 64e-5 32e-5
#do
#    python run_fed_avg.py --n_global_rounds 20000 --dataset mnist  --homo_ratio .1 --step_size_0 $j --p 0.1 --local_mb_size 256 --comm_max 2000 --worker_local_steps $i
#done
#done
