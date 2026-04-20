"""
N-CMAPSS Dataset loader for RUL prediction
Aligned with RESS 2022 Paper: "Fusing physics-based and deep learning models for prognostics"
Features: 10x Downsampling (50 points = 500s), Stride=1, Window=50, [-1,1] Normalization
"""
import os
import random
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
import torch

class NCMAPSSDataset:
    def __init__(self, 
                 data_root,
                 dataset_name='N-CMAPSS_DS02-006',
                 max_rul=125,
                 seq_len=5,
                 time_denpen_len=10,   # 5 * 10 = 50 (匹配窗口大小 50)
                 window_sample=50,     # 窗口大小设为 50 (结合降采样，代表物理时间 500 秒)
                 val_split=0.2):       # 恢复为 20% 的验证集比例
        """
        Args:
            data_root: Root directory containing NCMAPSSData folder
            dataset_name: N-CMAPSS dataset name
            max_rul: Maximum RUL value for clipping
            seq_len: Patch size for each time segment
            time_denpen_len: Number of time patches
            window_sample: Total window size (seq_len * time_denpen_len)
            val_split: Validation split ratio
        """
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.max_rul = max_rul
        self.seq_len = seq_len
        self.time_denpen_len = time_denpen_len
        self.window_sample = window_sample
        self.val_split = val_split
        self.downsample_rate = 10      # 新增: 10 倍降采样 (1Hz -> 0.1Hz)
        
        # N-CMAPSS feature info
        self.num_operating_conditions = 4
        self.num_sensors = 14
        self.total_features = self.num_operating_conditions + self.num_sensors  # 18
        
        # Load and process data
        self._load_data()
        self._process_data()
        self._create_splits()

    def _load_data(self):
        """Load data from N-CMAPSS h5 file"""
        if 'NCMAPSSData' in self.data_root:
            data_dir = self.data_root
        else:
            data_dir = os.path.join(self.data_root, 'NCMAPSSData')
        
        h5_path = os.path.join(data_dir, f'{self.dataset_name}.h5')
        print(f"[DATA] Loading N-CMAPSS dataset from: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            # Load development (training) data
            W_dev = f['W_dev'][:]           
            X_s_dev = f['X_s_dev'][:]       
            Y_dev = f['Y_dev'][:]           
            A_dev = f['A_dev'][:]           
            
            # Load test data
            W_test = f['W_test'][:]         
            X_s_test = f['X_s_test'][:]     
            Y_test = f['Y_test'][:]         
            A_test = f['A_test'][:]          
            
            # Extract unit IDs
            self.train_unit_ids = A_dev[:, 0].astype(int)
            self.test_unit_ids = A_test[:, 0].astype(int)
            
            self.train_units = np.unique(self.train_unit_ids)
            self.test_units = np.unique(self.test_unit_ids)
            
            self.W_dev = W_dev  
            self.X_s_dev = X_s_dev  
            self.Y_dev = Y_dev.flatten()  
            
            self.W_test = W_test
            self.X_s_test = X_s_test
            self.Y_test = Y_test.flatten()

    def _process_data(self):
        """Process and normalize data"""
        # Combine features: [W, X_s] -> [N, 18]
        self.train_features = np.concatenate([self.W_dev, self.X_s_dev], axis=1)
        self.test_features = np.concatenate([self.W_test, self.X_s_test], axis=1)
        
        # 按照论文要求，使用最小-最大归一化，将范围映射到 [-1, 1]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train_features = self.scaler.fit_transform(self.train_features)
        self.test_features = self.scaler.transform(self.test_features)
        
        print(f"[DATA] Normalization complete (Range: [-1, 1])")
        
        # Clip RUL to max_rul
        self.train_y = np.clip(self.Y_dev, 0, self.max_rul)
        self.test_y = np.clip(self.Y_test, 0, self.max_rul)
        
        # Generate sequences
        self._generate_sequences()

    def _generate_sequences(self):
        """Generate sliding window sequences for training and test"""
        print(f"[DATA] Applying {self.downsample_rate}x downsampling...")
        print(f"[DATA] Generating sequences with window_sample={self.window_sample} (representing {self.window_sample * self.downsample_rate} seconds)")
        
        self.train_x, self.train_ops, self.train_y_seq, self.train_engine_ids = self._create_train_sequences()
        self.test_x, self.test_ops, self.test_y_seq, self.test_engine_ids = self._create_test_sequences()
        
        print(f"[DATA] Generated {len(self.train_x)} training samples")
        print(f"[DATA] Generated {len(self.test_x)} test samples")

    def _create_train_sequences(self):
        """Create training sequences using sliding window over downsampled data"""
        seq_gen_x = []
        seq_gen_ops = []
        seq_gen_y = []
        seq_gen_engine_ids = []
        
        # 降采样后的滑窗步长设为 1，完全对齐论文
        stride = 1  
        
        for unit_id in self.train_units:
            mask = self.train_unit_ids == unit_id
            
            # 在单台发动机内部进行 10 倍降采样，避免跨发动机数据污染
            unit_features = self.train_features[mask][::self.downsample_rate]
            unit_y = self.train_y[mask][::self.downsample_rate]
            unit_len = len(unit_features)
            
            for start_idx in range(0, unit_len - self.window_sample + 1, stride):
                end_idx = start_idx + self.window_sample
                window_features = unit_features[start_idx:end_idx]
                window_y = unit_y[end_idx - 1] 
                
                x_patches, ops_patches = self._patch_sample(window_features)
                
                seq_gen_x.append(x_patches)
                seq_gen_ops.append(ops_patches)
                seq_gen_y.append(window_y)
                seq_gen_engine_ids.append(unit_id)
        
        return (np.array(seq_gen_x, dtype=np.float32),
                np.array(seq_gen_ops, dtype=np.float32),
                np.array(seq_gen_y, dtype=np.float32),
                np.array(seq_gen_engine_ids))

    def _create_test_sequences(self):
        """Create test sequences (evaluate FULL trajectory to expose true RMSE)"""
        seq_gen_x = []
        seq_gen_ops = []
        seq_gen_y = []
        seq_gen_engine_ids = []
        
        # 【核心修改】：测试集滑窗步长设为 10（比训练集快 10 倍），这样既能评估全生命周期，又能显著加快测试阶段的推理速度
        stride = 10 
        
        for unit_id in self.test_units:
            mask = self.test_unit_ids == unit_id
            
            # 测试集同样进行按单元的 10 倍降采样
            unit_features = self.test_features[mask][::self.downsample_rate]
            unit_y = self.test_y[mask][::self.downsample_rate]
            unit_len = len(unit_features)
            
            # 【核心修改】：滑动窗口遍历整台发动机的生命周期，而不再是只取最后一段！
            for start_idx in range(0, unit_len - self.window_sample + 1, stride):
                end_idx = start_idx + self.window_sample
                window_features = unit_features[start_idx:end_idx]
                window_y = unit_y[end_idx - 1] 
                
                x_patches, ops_patches = self._patch_sample(window_features)
                
                seq_gen_x.append(x_patches)
                seq_gen_ops.append(ops_patches)
                seq_gen_y.append(window_y)
                seq_gen_engine_ids.append(unit_id)
        
        return (np.array(seq_gen_x, dtype=np.float32),
                np.array(seq_gen_ops, dtype=np.float32),
                np.array(seq_gen_y, dtype=np.float32),
                np.array(seq_gen_engine_ids))

    def _patch_sample(self, window_features):
        """Divide window into patches for the model"""
        patch_size = self.seq_len
        num_patches = self.time_denpen_len
        expected_window = patch_size * num_patches
        
        if len(window_features) > expected_window:
            window_features = window_features[:expected_window]
        elif len(window_features) < expected_window:
            pad_len = expected_window - len(window_features)
            window_features = np.vstack([
                window_features,
                np.tile(window_features[-1], (pad_len, 1))
            ])
        
        x_patches = [] 
        ops_patches = [] 
        
        for i in range(num_patches):
            start = i * patch_size
            end = start + patch_size
            patch = window_features[start:end]
            
            ops_patch = patch[:, :self.num_operating_conditions]  
            sensor_patch = patch[:, self.num_operating_conditions:] 
            
            x_patches.append(sensor_patch)
            ops_patches.append(ops_patch)
        
        return np.array(x_patches, dtype=np.float32), np.array(ops_patches, dtype=np.float32)

    def _create_splits(self):
        """Create train/validation splits"""
        num_samples = len(self.train_x)
        indices = list(range(num_samples))
        random.shuffle(indices)
        
        # 恢复为传入的 val_split (0.2)
        val_size = int(num_samples * self.val_split)
        self.val_indices = indices[:val_size]
        self.train_indices = indices[val_size:]

    def get_train_data(self):
        """返回传感器数据 (丢弃前4列工况信号)"""
        combined_x = np.concatenate([self.train_ops[self.train_indices], self.train_x[self.train_indices]], axis=-1)
        return {
            'x': combined_x[..., 4:],  # 切掉前4列工况信号，只保留14个传感器
            'y': self.train_y_seq[self.train_indices] / self.max_rul
        }

    def get_val_data(self):
        """返回传感器数据 (丢弃前4列工况信号)"""
        # 验证集使用 train_x 和 train_ops 的 val_indices 部分
        val_x = self.train_x[self.val_indices]
        return {
            'x': val_x,  # train_x 已经是14个传感器，不需要切片
            'y': self.train_y_seq[self.val_indices] / self.max_rul,
            'engine_id': self.train_engine_ids[self.val_indices]
        }

    def get_test_data(self):
        """返回传感器数据 (丢弃前4列工况信号)"""
        return {
            'x': self.test_x,  # test_x 已经是14个传感器
            'y': self.test_y_seq / self.max_rul
        }

    def get_train_data(self):
        """返回传感器数据 (丢弃前4列工况信号)"""
        train_x = self.train_x[self.train_indices]
        return {
            'x': train_x,  # train_x 已经是14个传感器
            'y': self.train_y_seq[self.train_indices] / self.max_rul
        }

    @property
    def dataset_name_full(self):
        return self.dataset_name