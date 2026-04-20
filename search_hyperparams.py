import optuna
import optunahub
import sys
import os
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_single_experiment

# 加载 OptunaHub 的高级Sampler（支持多目标）
try:
    sampler_module = optunahub.load_module(package="simplicial-hypervolume")
    sampler = sampler_module.Sampler()
    print("[SEARCH] Using Simplicial-Hypervolume Sampler")
except Exception as e:
    print(f"[SEARCH] Fallback to default TPE Sampler: {e}")
    sampler = optuna.samplers.TPESampler()

def objective(trial: optuna.Trial):
    config = {
        "dataset_name": "N-CMAPSS_DS02-006",
        "data_params": {
            "window_sample": trial.suggest_categorical("window_sample", [30, 50, 80, 150]), # 探索更长窗口
            "patch_size": 5, # 固定
            "max_rul": 125,
        },
        "model_params": {
            "num_sensor": 14, # 锁死纯传感器模式
            "num_tokens": trial.suggest_int("num_tokens", 16, 64, step=16),
            "token_dim": trial.suggest_categorical("token_dim", [64, 128, 256]),
            # 其他模型参数固定为默认值
        },
        "training_params": {
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True), # 锁定在稳健区间
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True), # 重点搜索正则化
            "epochs": 10,
        },
    }
    
    # 手动计算对齐 time_denpen_len 避免报错
    config["data_params"]["time_denpen_len"] = config["data_params"]["window_sample"] // config["data_params"]["patch_size"]

    try:
        rmse = run_single_experiment(config)
        return rmse
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")

# 启动优化
study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    study_name="ncmapss_phasepool_search",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
)

# 加速：允许并发（需要 optuna + joblib）
study.optimize(
    objective,
    n_trials=50,
    timeout=3600 * 8,  # 最多运行8小时
    n_jobs=1,          # 设为2可加速，但显存需足够
    show_progress_bar=True,
)

# 保存最佳参数
best_params = {
    "best_rmse": study.best_value,
    "best_params": study.best_params,
}
with open("best_hyperparams.json", "w") as f:
    json.dump(best_params, f, indent=2)

print(f"\n[SEARCH] Best RMSE: {study.best_value:.4f}")
print(f"[SEARCH] Best params: {study.best_params}")