import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from dataset_ncmapss import NCMAPSSDataset
from models.phasepool_tokenmixer import PhasePoolTokenMixerRUL

# ==========================================
# 1. ????
# ==========================================
CHECKPOINT_PATH = r"C:\Users\hxq\Desktop\auto\phasepool\NCMAPPSS\PhasePool-TokenMixer\experiments\runs\ncmapss_baseline_20260420_132240\checkpoints\best_model_epoch003_score1.pth"
DATA_ROOT = "./"
TARGET_UNIT_ID = 11

# ==========================================
# 2. ??? checkpoint ???? config.json
# ==========================================
def load_config_from_checkpoint(checkpoint_path):
    """? checkpoint ?????????? config.json"""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, '..', 'config', 'config.json')
    config_path = os.path.normpath(config_path)
    print(f"[CONFIG] Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# ????
config = load_config_from_checkpoint(CHECKPOINT_PATH)

# ??????
DATA_PARAMS = config['data_params']
MODEL_PARAMS = config['model_params']

WINDOW_SAMPLE = DATA_PARAMS['window_sample']
PATCH_SIZE = DATA_PARAMS['patch_size']
TIME_DEPEN_LEN = DATA_PARAMS['time_denpen_len']
MAX_RUL = DATA_PARAMS['max_rul']

NUM_TOKENS = MODEL_PARAMS['num_tokens']
TOKEN_DIM = MODEL_PARAMS['token_dim']
NUM_HEADS = MODEL_PARAMS['num_heads']
TEMPERATURE = MODEL_PARAMS['temperature']
USE_POS_ENCODING = MODEL_PARAMS['use_pos_encoding']

print(f"[CONFIG] Model Params:")
print(f"  - patch_size: {PATCH_SIZE}")
print(f"  - time_denpen_len: {TIME_DEPEN_LEN}")
print(f"  - num_tokens: {NUM_TOKENS}")
print(f"  - token_dim: {TOKEN_DIM}")
print(f"  - num_heads: {NUM_HEADS}")
print(f"  - temperature: {TEMPERATURE}")
print(f"  - use_pos_encoding: {USE_POS_ENCODING}")

DOWNSAMPLE_RATE = 10

# ==========================================
# 3. ????
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 4. ?????
# ==========================================
print("[TEST] Loading Dataset...")
dataset = NCMAPSSDataset(
    data_root=DATA_ROOT,
    dataset_name='N-CMAPSS_DS02-006',
    max_rul=MAX_RUL,
    seq_len=PATCH_SIZE,
    time_denpen_len=TIME_DEPEN_LEN,
    window_sample=WINDOW_SAMPLE
)

# ==========================================
# 5. ????
# ==========================================
print("[TEST] Building Model...")
model = PhasePoolTokenMixerRUL(
    patch_size=PATCH_SIZE,
    time_denpen_len=TIME_DEPEN_LEN,
    num_sensor=14,  # 改为14，移除4个工况信号
    num_tokens=NUM_TOKENS,
    token_dim=TOKEN_DIM,
    num_heads=NUM_HEADS,
    temperature=TEMPERATURE,
    pe_on_input=USE_POS_ENCODING
).to(device)

print(f"[TEST] Loading Checkpoint from: {CHECKPOINT_PATH}")
state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
if 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']

# ????
try:
    model.load_state_dict(state_dict, strict=True)
    print("[TEST] Checkpoint loaded successfully (strict=True)")
except Exception as e:
    print(f"[TEST] Warning: {e}")
    model.load_state_dict(state_dict, strict=False)

model.eval()

# ==========================================
# 6. ??????
# ==========================================
print(f"[TEST] Extracting trajectory for Test Unit {TARGET_UNIT_ID}...")

h5_path = os.path.join(DATA_ROOT, 'NCMAPSSData', 'N-CMAPSS_DS02-006.h5')
with h5py.File(h5_path, 'r') as f:
    W_test = f['W_test'][:]
    X_s_test = f['X_s_test'][:]
    Y_test = f['Y_test'][:]
    A_test = f['A_test'][:]
    test_unit_ids = A_test[:, 0].astype(int)
    mask = test_unit_ids == TARGET_UNIT_ID
    unit_features = np.concatenate([W_test[mask], X_s_test[mask]], axis=1)
    unit_y = np.clip(Y_test[mask].flatten(), 0, MAX_RUL)
    unit_features = unit_features[::DOWNSAMPLE_RATE]
    unit_y = unit_y[::DOWNSAMPLE_RATE]
    unit_len = len(unit_features)

print(f"[TEST] Unit {TARGET_UNIT_ID} has {unit_len} downsampled time steps")

# ??
predictions = []
actual_ruls = []
total_steps = unit_len - WINDOW_SAMPLE + 1
print(f"[TEST] Running inference on {total_steps} windows...")

with torch.no_grad():
    for i, start_idx in enumerate(range(0, unit_len - WINDOW_SAMPLE + 1, 1)):
        end_idx = start_idx + WINDOW_SAMPLE
        window_features = unit_features[start_idx:end_idx]
        true_rul = unit_y[end_idx - 1]
        actual_ruls.append(true_rul)
        x_patches, _ = dataset._patch_sample(window_features)
        input_tensor = torch.FloatTensor(x_patches).unsqueeze(0).to(device)
        pred_rul = model(input_tensor).item() * MAX_RUL
        predictions.append(pred_rul)
        if (i + 1) % 5000 == 0 or (i + 1) == total_steps:
            print(f"  Progress: {i + 1}/{total_steps} ({100 * (i + 1) / total_steps:.1f}%)")

# ==========================================
# 7. ??
# ==========================================
print("[TEST] Plotting Results...")
plt.figure(figsize=(12, 6))
plt.plot(actual_ruls, label='True RUL', color='black', linewidth=2)
plt.plot(predictions, label='PhasePool-TokenMixer Prediction', color='blue', alpha=0.8)
plt.axhline(y=MAX_RUL, color='red', linestyle='--', alpha=0.5, label='Max RUL Threshold')
plt.title(f'RUL Prediction Trajectory - N-CMAPSS DS02 (Unit {TARGET_UNIT_ID})', fontsize=14)
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('RUL', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

checkpoint_name = os.path.basename(os.path.dirname(os.path.dirname(CHECKPOINT_PATH)))
save_path = f"unit_{TARGET_UNIT_ID}_{checkpoint_name}_prediction_curve.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
plt.show()

# ????
predictions = np.array(predictions)
actual_ruls = np.array(actual_ruls)
rmse = np.sqrt(np.mean((predictions - actual_ruls) ** 2))
mae = np.mean(np.abs(predictions - actual_ruls))
print(f"\n[TEST] Metrics:")
print(f"  - RMSE: {rmse:.2f}")
print(f"  - MAE: {mae:.2f}")