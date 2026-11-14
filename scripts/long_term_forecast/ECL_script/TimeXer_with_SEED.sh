
#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

model_name=TimeXer_with_SEED

# 実験ごとに異なるログ保存先を作成
log_dir="logs/${model_name}_ECL_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# 実験を実行する関数
run_exp() {
    local seed=$1
    local pred_len=$2
    local e_layers=$3
    local d_ff=${4:-}

    local log_file="${log_dir}/pred${pred_len}_seed${seed}.log"

    echo "--------------------------------------"
    echo "seed=${seed}, pred_len=${pred_len}"
    echo "--------------------------------------"

    args=(
        --task_name long_term_forecast
        --is_training 1
        --root_path ../dataset/electricity/
        --data_path electricity.csv
        --model_id ECL_96_${pred_len}
        --model "$model_name"
        --data custom
        --features M
        --seq_len 96
        --label_len 48
        --pred_len "$pred_len"
        --e_layers "$e_layers"
        --factor 3
        --enc_in 321
        --dec_in 321
        --c_out 321
        --batch_size 4
        --des "seed${seed}"
        --random_seed "$seed"
        --itr 1
    )

    # d_ffが指定されている場合だけ追加
    if [[ -n "$d_ff" ]]; then
        args+=(--d_ff "$d_ff")
    fi

    # 実行ログを保存しながら画面にも表示
    python -u run.py "${args[@]}" 2>&1 | tee "$log_file"
}

# 5つの乱数シードで実験
for seed in 2022 2023 2024 2025 2026
do

    echo "======================================"
    echo "Running experiments with seed: $seed"
    echo "======================================"

    # pred_len = 96
    run_exp "$seed" 96 4 512

    # pred_len = 192
    run_exp "$seed" 192 3

    # pred_len = 336
    run_exp "$seed" 336 4

    # pred_len = 720
    run_exp "$seed" 720 3

done

# --------------------------------------
# 実験終了後に結果を集計
# --------------------------------------

LOG_DIR="$log_dir" python - <<'PY'
import os
import re
import glob
import csv
import numpy as np

log_dir = os.environ["LOG_DIR"]

results = []

# 各ログファイルからMSEとMAEを取得
for file in sorted(glob.glob(os.path.join(log_dir, "*.log"))):

    filename = os.path.basename(file)

    match = re.search(r"pred(\d+)_seed(\d+)", filename)

    if match is None:
        continue

    pred_len = int(match.group(1))
    seed = int(match.group(2))

    mse = None
    mae = None

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            mse_match = re.search(
                r"mse\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
                line,
                re.IGNORECASE
            )

            mae_match = re.search(
                r"mae\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
                line,
                re.IGNORECASE
            )

            if mse_match and mae_match:
                mse = float(mse_match.group(1))
                mae = float(mae_match.group(1))

    if mse is None or mae is None:
        print(f"WARNING: MSE/MAEを取得できませんでした: {filename}")
        continue

    results.append({
        "pred_len": pred_len,
        "seed": seed,
        "MSE": mse,
        "MAE": mae
    })

# 各シードの結果をCSVに保存
detail_path = os.path.join(log_dir, "seed_results.csv")

with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["pred_len", "seed", "MSE", "MAE"]
    )
    writer.writeheader()
    writer.writerows(results)

# 予測長ごとに平均と標準偏差を計算
summary = []

for pred_len in [96, 192, 336, 720]:

    group = [
        r for r in results
        if r["pred_len"] == pred_len
    ]

    if not group:
        continue

    mse_values = np.array([r["MSE"] for r in group])
    mae_values = np.array([r["MAE"] for r in group])

    summary.append({
        "pred_len": pred_len,
        "n_seeds": len(group),
        "MSE_mean": mse_values.mean(),
        "MSE_std": mse_values.std(ddof=1) if len(group) > 1 else np.nan,
        "MAE_mean": mae_values.mean(),
        "MAE_std": mae_values.std(ddof=1) if len(group) > 1 else np.nan
    })

summary_path = os.path.join(log_dir, "seed_summary.csv")

with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "pred_len",
            "n_seeds",
            "MSE_mean",
            "MSE_std",
            "MAE_mean",
            "MAE_std"
        ]
    )
    writer.writeheader()
    writer.writerows(summary)

# 結果を画面にも表示
print("\n======================================")
print("       実験結果の集計")
print("======================================")

for r in summary:
    print(
        f"pred_len={r['pred_len']}: "
        f"MSE={r['MSE_mean']:.6f} ± {r['MSE_std']:.6f}, "
        f"MAE={r['MAE_mean']:.6f} ± {r['MAE_std']:.6f}, "
        f"n={r['n_seeds']}"
    )

print("\n保存先:")
print(detail_path)
print(summary_path)

PY
