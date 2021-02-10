#for eta in 30 35 40
#do
#for K in 1 2 3
#do
#for p in .75 .8 .85 .9 .95 1
#do
#    python run_ffgd.py --n_global_rounds 20000 --dataset mnist  --homo_ratio .3 --step_size_0 $eta\
#     --p $p --oracle_mb_size 128 --comm_max 2100 --worker_local_steps $K --n_workers 56 --n_ray_workers 56 \
#     --oracle_step_size 5e-3 --oracle_local_steps 4000 --device cpu --backend joblib
#done
#done
#done



#for eta in 30 35 40
#do
#for K in 10 15 20
#do
#for p in 1
#do
#python run_ffgd.py --n_global_rounds 101 --dataset mnist  --homo_ratio .1 --step_size_0 $eta\
# --p $p --oracle_mb_size 128 --comm_max 0 --worker_local_steps $K --n_workers 56 --n_ray_workers 56 \
# --oracle_step_size 5e-3 --oracle_local_steps 2000 --device cpu --backend joblib --weak_learner_hid_dims 32-32
#done
#done
#done

#for eta in 30 35 40
#do
#for s in 0.1
#do
#python run_ffgd.py --n_global_rounds 101 --dataset mnist  --homo_ratio $s --step_size_0 20\
# --p 1 --oracle_mb_size 128 --comm_max 0 --worker_local_steps 2 --n_workers 56 --n_ray_workers 56 \
# --oracle_step_size 5e-3 --oracle_local_steps 2000 --device cpu --backend joblib --weak_learner_hid_dims 32-32\
# --use_adv_label True
#done
for i in 1 2 3 4 5
do
for s in 0.1 0.3
do
python run_ffgd.py --n_global_rounds 101 --dataset mnist  --homo_ratio $s --step_size_0 20\
 --p 1 --oracle_mb_size 128 --comm_max 2000 --worker_local_steps 2 --n_workers 56 --n_ray_workers 4 \
 --oracle_step_size 5e-3 --oracle_local_steps 2000 --device cpu --backend joblib --weak_learner_hid_dims 32-32
done
done

