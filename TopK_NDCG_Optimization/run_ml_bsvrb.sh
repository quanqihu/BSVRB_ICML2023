gpu_id_a=0
loss_type=BSVRB

python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 0.0004 --l2 1e-7 --num_neg 300 --num_pos 10 \
               --dropout 0.2 --dataset 'ml-20m' --batch_size 256  --eval_batch_size 512 \
               --loss_type ${loss_type} \
               --metric NDCG \
               --ndcg_gamma 0.1 --ndcg_topk 300 \
               --gamma_v 0.9 --beta_v 0.005 \
               --gamma_s 0.9 --beta_s 0.005 \
               --gamma_z 0.9 --beta_z 0.005 \
               --gamma_m 0.01 --eta1 0.4 \
               --load 1 --init_last 1 \
               --pretrain_model ../model/NeuMF/ml-20m_pretrained.pt \
               --gpu ${gpu_id_a} --reorg_train_data 1 \
               --epoch 120 \
               --random_seed 0 \
               --run_name ml-20m_${loss_type}_baseline


















