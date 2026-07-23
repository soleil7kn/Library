#!/bin/bash

export PYTHONHASHSEED=2026
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer_iTransformer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16

# iTransformerの補正パラメータ
refiner_d_model=64
refiner_n_heads=4
refiner_d_ff=128
refiner_e_layers=1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_$seq_len'_'96\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des Exp \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --num_workers 0 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --use_future_temporal_feature 0 \
  --refiner_d_model $refiner_d_model \
  --refiner_n_heads $refiner_n_heads \
  --refiner_d_ff $refiner_d_ff \
  --refiner_e_layers $refiner_e_layers \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_$seq_len'_'192\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers $e_layers \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des Exp \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --num_workers 0 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --use_future_temporal_feature 0 \
  --refiner_d_model $refiner_d_model \
  --refiner_n_heads $refiner_n_heads \
  --refiner_d_ff $refiner_d_ff \
  --refiner_e_layers $refiner_e_layers \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_$seq_len'_'336\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des Exp \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --num_workers 0 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --use_future_temporal_feature 0 \
  --refiner_d_model $refiner_d_model \
  --refiner_n_heads $refiner_n_heads \
  --refiner_d_ff $refiner_d_ff \
  --refiner_e_layers $refiner_e_layers \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_$seq_len'_'720\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des Exp \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --num_workers 0 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --use_future_temporal_feature 0 \
  --refiner_d_model $refiner_d_model \
  --refiner_n_heads $refiner_n_heads \
  --refiner_d_ff $refiner_d_ff \
  --refiner_e_layers $refiner_e_layers \