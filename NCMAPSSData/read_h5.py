import h5py
import numpy as np

file_path = r'C:\Users\hxq\Desktop\auto\phasepool\NCMAPPSS\PhasePool-TokenMixer\NCMAPSSData\N-CMAPSS_DS02-006.h5'

with h5py.File(file_path, 'r') as f:
    print("✅ 数据集读取成功，开始数据切分...")
    
    # 1. 提取辅助信息 A_dev。它的第一列 (index 0) 就是发动机编号 (Unit ID)
    A_dev = f['A_dev'][:]
    
    # 我们先拿编号为 2 的发动机 (Unit 2) 来做单机测试
    target_unit = 2.0
    unit_mask = (A_dev[:, 0] == target_unit) # 生成布尔掩码，挑出属于 Unit 2 的行
    
    # 2. 根据掩码，将 Unit 2 的特征和标签剥离出来
    W_unit = f['W_dev'][unit_mask]
    Xs_unit = f['X_s_dev'][unit_mask]
    Y_unit = f['Y_dev'][unit_mask]
    
    print(f"\n--- 发动机 Unit {target_unit} 独立数据 ---")
    print(f"分离后工况 W 维度: {W_unit.shape}") 
    print(f"分离后特征 Xs 维度: {Xs_unit.shape}")
    
    # 3. 模拟深度学习中的滑动窗口 (Sliding Window)
    # 将 2D 序列转换为 3D 张量 [Samples, Window_Size, Features]
    window_size = 50  # 你的 TokenMixer 每次能看到的序列长度
    stride = 10       # 步长设为10，跳跃采样以减小数据量和相邻样本的冗余度
    
    # 简单拼接：把工况 W (4维) 和 传感器 Xs (14维) 拼成 18 维特征
    features_combined = np.concatenate((W_unit, Xs_unit), axis=1)
    
    windows_X = []
    windows_Y = []
    
    # 滑窗过程
    for i in range(0, len(features_combined) - window_size, stride):
        # 框出 window_size 长度的特征
        windows_X.append(features_combined[i : i + window_size])
        # RUL 预测通常取窗口最后一个时间点的寿命作为当前窗口的标签
        windows_Y.append(Y_unit[i + window_size - 1])
        
    windows_X = np.array(windows_X)
    windows_Y = np.array(windows_Y)
    
    print(f"\n--- 送入神经网络的张量准备完毕 ---")
    print(f"X 张量维度: {windows_X.shape}  -> [样本数, 序列长度(Tokens), 特征维度]")
    print(f"Y 标签维度: {windows_Y.shape}  -> [样本数, 1]")