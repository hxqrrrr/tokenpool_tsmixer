"""
CMAPSS Dataset loader for RUL prediction
Refactored from data_loader_RUL.py for better modularity
"""
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch


class CMAPSSDataset:
    """
    Dataset class for CMAPSS (C-MAPSS) turbofan engine degradation dataset
    """
    
    def __init__(self, 
                 data_root,
                 dataset_name='FD001',
                 max_rul=125,
                 seq_len=5,
                 time_denpen_len=6,
                 window_sample=30,
                 val_split=0.2):
        """
        Args:
            data_root: Root directory containing CMAPSSData folder
            dataset_name: Dataset name (FD001, FD002, FD003, FD004)
            max_rul: Maximum RUL value for clipping
            seq_len: Patch size for each time segment
            time_denpen_len: Number of time patches
            window_sample: Total window size (seq_len * time_denpen_len)
            val_split: Validation split ratio (default 0.2 for 5-fold CV)
        """
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.max_rul = max_rul
        self.seq_len = seq_len
        self.time_denpen_len = time_denpen_len
        self.window_sample = window_sample
        self.val_split = val_split
        
        self.column_names = [
            'id', 'cycle', 'setting1', 'setting2', 'setting3',
            's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
            's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
        ]
        
        # Load and process data
        self._load_data()
        self._process_data()
        self._create_splits()
        
    def _load_data(self):
        """Load raw data files"""
        data_dir = os.path.join(self.data_root, 'CMAPSSData')
        
        # Load training data
        train_path = os.path.join(data_dir, f'train_{self.dataset_name}.txt')
        self.train_df = pd.read_csv(train_path, sep=" ", header=None)
        self.train_df.drop(self.train_df.columns[[26, 27]], axis=1, inplace=True)
        self.train_df.columns = self.column_names
        self.train_df = self.train_df.sort_values(['id', 'cycle'])
        
        # Load testing data
        test_path = os.path.join(data_dir, f'test_{self.dataset_name}.txt')
        self.test_df = pd.read_csv(test_path, sep=" ", header=None)
        self.test_df.drop(self.test_df.columns[[26, 27]], axis=1, inplace=True)
        self.test_df.columns = self.column_names
        self.test_df = self.test_df.sort_values(['id', 'cycle'])
        
        # Load test truth (remaining RUL)
        truth_path = os.path.join(data_dir, f'RUL_{self.dataset_name}.txt')
        self.test_truth = pd.read_csv(truth_path, sep=" ", header=None)
        self.test_truth.drop(self.test_truth.columns[[1]], axis=1, inplace=True)
        
    def _process_data(self):
        """Process and normalize data"""
        # Add RUL labels to training data
        train_rul = pd.DataFrame(self.train_df.groupby('id')['cycle'].max()).reset_index()
        train_rul.columns = ['id', 'max']
        self.train_df = self.train_df.merge(train_rul, on=['id'], how='left')
        train_y = pd.DataFrame(data=[self.train_df['max'] - self.train_df['cycle']]).T
        self.train_df.drop('max', axis=1, inplace=True)
        
        # Drop constant sensors
        self.train_df.drop(['s1', 's5', 's6', 's10', 's16', 's18', 's19'], axis=1, inplace=True)
        self.train_df['setting1'] = self.train_df['setting1'].round(1)
        
        # Clip RUL
        train_y = train_y.apply(lambda x: [y if y <= self.max_rul else self.max_rul for y in x])
        
        # Process test data
        test_rul = pd.DataFrame(self.test_df.groupby('id')['cycle'].max()).reset_index()
        test_rul.columns = ['id', 'max']
        
        self.test_truth.columns = ['more']
        self.test_truth['id'] = self.test_truth.index + 1
        self.test_truth['max'] = test_rul['max'] + self.test_truth['more']
        self.test_truth.drop('more', axis=1, inplace=True)
        
        self.test_df = self.test_df.merge(self.test_truth, on=['id'], how='left')
        test_y = pd.DataFrame(data=[self.test_df['max'] - self.test_df['cycle']]).T
        self.test_df.drop('max', axis=1, inplace=True)
        
        # Drop constant sensors from test
        self.test_df.drop(['s1', 's5', 's6', 's10', 's16', 's18', 's19'], axis=1, inplace=True)
        self.test_df['setting1'] = self.test_df['setting1'].round(1)
        
        # Clip test RUL
        test_y = test_y.apply(lambda x: [y if y <= self.max_rul else self.max_rul for y in x])
        
        # Normalize data
        self._normalize_data()
        
        # Generate sequences
        self.train_x, self.train_ops, self.train_y = self._generate_sequences(
            self.train_normalized, self.train_setting, train_y, self.train_df, is_train=True
        )
        
        self.test_x, self.test_ops, self.test_y = self._generate_test_sequences(
            self.test_normalized, self.test_setting, test_y, self.test_df
        )
        
    def _normalize_data(self):
        """Normalize sensor readings and settings"""
        train_data = self.train_df.iloc[:, 2:]
        test_data = self.test_df.iloc[:, 2:]
        
        self.train_normalized = pd.DataFrame(columns=train_data.columns[3:])
        self.test_normalized = pd.DataFrame(columns=test_data.columns[3:])
        
        scaler = MinMaxScaler()
        
        # Normalize by operating condition (setting1)
        for train_idx, train_group in train_data.groupby('setting1'):
            scaled_train = scaler.fit_transform(train_group.iloc[:, 3:])
            scaled_train_df = pd.DataFrame(
                data=scaled_train,
                index=train_group.index,
                columns=train_data.columns[3:]
            )
            self.train_normalized = pd.concat([self.train_normalized, scaled_train_df])
            
            # Apply same scaling to test data with same operating condition
            for test_idx, test_group in test_data.groupby('setting1'):
                if train_idx == test_idx:
                    scaled_test = scaler.transform(test_group.iloc[:, 3:])
                    scaled_test_df = pd.DataFrame(
                        data=scaled_test,
                        index=test_group.index,
                        columns=test_data.columns[3:]
                    )
                    self.test_normalized = pd.concat([self.test_normalized, scaled_test_df])
        
        self.train_normalized = self.train_normalized.sort_index()
        self.test_normalized = self.test_normalized.sort_index()
        
        # Normalize settings
        scaler = MinMaxScaler()
        self.train_setting = pd.DataFrame(
            data=scaler.fit_transform(self.train_df.iloc[:, 1:5]),
            index=self.train_df.index,
            columns=self.train_df.columns[1:5]
        )
        self.test_setting = pd.DataFrame(
            data=scaler.transform(self.test_df.iloc[:, 1:5]),
            index=self.test_df.index,
            columns=self.test_df.columns[1:5]
        )
        
    def _generate_sequences(self, sensor_data, setting_data, labels, metadata, is_train=True):
        """Generate sliding window sequences"""
        engine_ids = metadata['id'].unique()
        num_engines = len(engine_ids)
        
        seq_gen_x = []
        seq_gen_ops = []
        seq_gen_y = []
        
        start_index = 0
        for engine_id in engine_ids:
            engine_data = metadata[metadata['id'] == engine_id]
            end_index = start_index + len(engine_data)
            
            # Generate sensor sequences
            sensor_seq = list(self._gen_sequence(
                sensor_data.iloc[start_index:end_index, :],
                self.window_sample,
                sensor_data.columns
            ))
            seq_gen_x.extend(sensor_seq)
            
            # Generate setting sequences
            setting_seq = list(self._gen_sequence(
                setting_data.iloc[start_index:end_index, :],
                self.window_sample,
                setting_data.columns
            ))
            seq_gen_ops.extend(setting_seq)
            
            # Generate labels
            label_seq = list(self._gen_labels(
                labels.iloc[start_index:end_index, :],
                self.window_sample,
                labels.columns
            ))
            seq_gen_y.extend(label_seq)
            
            start_index = end_index
        
        # Apply patch sampling
        x_data = self._data_sampling(np.array(seq_gen_x), self.seq_len, self.time_denpen_len)
        ops_data = self._data_sampling(np.array(seq_gen_ops), self.seq_len, self.time_denpen_len)
        y_data = np.array(seq_gen_y) / self.max_rul  # Normalize labels
        
        return x_data, ops_data, y_data
    
    def _generate_test_sequences(self, sensor_data, setting_data, labels, metadata):
        """Generate test sequences (only last window per engine)"""
        engine_ids = metadata['id'].unique()
        
        seq_gen_x = []
        seq_gen_ops = []
        seq_gen_y = []
        
        start_index = 0
        for engine_id in engine_ids:
            engine_data = metadata[metadata['id'] == engine_id]
            end_index = start_index + len(engine_data)
            engine_len = end_index - start_index
            
            # Handle sequences shorter than window_sample
            if engine_len < self.window_sample:
                # Pad with first data point
                num_pad = self.window_sample - engine_len
                sensor_seg = sensor_data.iloc[start_index:end_index, :]
                setting_seg = setting_data.iloc[start_index:end_index, :]
                
                for _ in range(num_pad):
                    sensor_seg = pd.concat([sensor_seg.head(1), sensor_seg], axis=0)
                    setting_seg = pd.concat([setting_seg.head(1), setting_seg], axis=0)
                
                sensor_seq = list(self._gen_sequence(sensor_seg, self.window_sample, sensor_data.columns))
                setting_seq = list(self._gen_sequence(setting_seg, self.window_sample, setting_data.columns))
            else:
                # Take last window
                sensor_seq = list(self._gen_sequence(
                    sensor_data.iloc[end_index - self.window_sample:end_index, :],
                    self.window_sample,
                    sensor_data.columns
                ))
                setting_seq = list(self._gen_sequence(
                    setting_data.iloc[end_index - self.window_sample:end_index, :],
                    self.window_sample,
                    setting_data.columns
                ))
            
            seq_gen_x.extend(sensor_seq)
            seq_gen_ops.extend(setting_seq)
            
            # Get last label
            seq_gen_y.append(labels.iloc[end_index - 1, 0])
            
            start_index = end_index
        
        # Apply patch sampling
        x_data = self._data_sampling(np.array(seq_gen_x), self.seq_len, self.time_denpen_len)
        ops_data = self._data_sampling(np.array(seq_gen_ops), self.seq_len, self.time_denpen_len)
        y_data = np.array(seq_gen_y) / self.max_rul  # Normalize labels
        
        return x_data, ops_data, y_data
    
    def _gen_sequence(self, df, seq_length, cols):
        """Generate sliding window sequences"""
        data_matrix = df[cols].values.astype(np.float32)
        num_elements = data_matrix.shape[0]
        
        for start, stop in zip(range(0, num_elements - seq_length + 1), 
                               range(seq_length, num_elements + 1)):
            yield data_matrix[start:stop, :]
    
    def _gen_labels(self, df, seq_length, label_cols):
        """Generate labels for sequences"""
        data_matrix = df[label_cols].values
        num_elements = data_matrix.shape[0]
        
        for i in range(num_elements - (seq_length - 1)):
            yield data_matrix[i + (seq_length - 1), 0]
    
    def _data_sampling(self, data, window_size, time_length):
        """Sample data into patches"""
        data_ls = []
        for i in range(time_length):
            data_i = data[:, i * window_size:(i + 1) * window_size, :]
            data_ls.append(data_i)
        
        return np.stack(data_ls, 1)
    
    def _create_splits(self):
        """Create train/validation splits using cross-validation"""
        num_samples = len(self.train_x)
        indices = list(range(num_samples))
        random.shuffle(indices)
        
        # 5-fold cross-validation split
        fold_size = num_samples // 5
        val_indices = indices[:fold_size]
        train_indices = indices[fold_size:]
        
        self.train_indices = train_indices
        self.val_indices = val_indices
    
    def get_train_data(self):
        """Get training data"""
        return {
            'x': self.train_x[self.train_indices],
            'ops': self.train_ops[self.train_indices],
            'y': self.train_y[self.train_indices]
        }
    
    def get_val_data(self):
        """Get validation data"""
        return {
            'x': self.train_x[self.val_indices],
            'ops': self.train_ops[self.val_indices],
            'y': self.train_y[self.val_indices]
        }
    
    def get_test_data(self):
        """Get test data"""
        return {
            'x': self.test_x,
            'ops': self.test_ops,
            'y': self.test_y
        }
    
    def get_all_train_data(self):
        """Get all training data (without validation split)"""
        return {
            'x': self.train_x,
            'ops': self.train_ops,
            'y': self.train_y
        }

