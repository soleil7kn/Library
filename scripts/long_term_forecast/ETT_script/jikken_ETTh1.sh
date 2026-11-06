
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

model_name=jikken

for seed in 2022 2023 2024 2025 2026
do
    echo "======================================"
    echo "Running experiments with seed: $seed"
    echo "======================================"

    # pred_len = 96
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96 \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --e_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 256 \
      --des "seed${seed}" \
      --random_seed $seed \
      --itr 1

    # pred_len = 192
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_192 \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 192 \
      --e_layers 2 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des "seed${seed}" \
      --d_model 128 \
      --random_seed $seed \
      --itr 1

    # pred_len = 336
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_336 \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 336 \
      --e_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des "seed${seed}" \
      --d_model 512 \
      --d_ff 1024 \
      --random_seed $seed \
      --itr 1

    # pred_len = 720
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ../dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_720 \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 720 \
      --e_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des "seed${seed}" \
      --d_model 256 \
      --d_ff 1024 \
      --random_seed $seed \
      --itr 1

done