"""
绘制预测误差分布图
包含直方图、正态分布拟合和KDE曲线

用法:
    python plot_error_distribution.py <predictions_csv> [--output OUTPUT]

示例:
    # 从predictions_detail.csv生成误差分布图
    python plot_error_distribution.py experiments/runs/baseline_fd002_20251022_125017/predictions_detail/predictions_detail.csv
    
    # 从batch_summary.csv生成误差分布图
    python plot_error_distribution.py experiments/runs/baseline_fd002_20251022_125017/batch_engines/train/batch_summary.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats
from scipy.stats import norm


def calculate_statistics(errors):
    """计算误差统计指标"""
    stats_dict = {
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors, ddof=1),
        'skewness': stats.skew(errors),
        'kurtosis': stats.kurtosis(errors),
        'range': np.max(errors) - np.min(errors),
        'min': np.min(errors),
        'max': np.max(errors)
    }
    return stats_dict


def plot_error_distribution(errors, output_path, title='Prediction Error Distribution',
                           figsize=(6, 5), dpi=300, n_bins=None):
    """
    绘制误差分布图（类似论文Fig. 8风格）
    
    Args:
        errors: 误差数组
        output_path: 输出文件路径
        title: 图表标题
        figsize: 图表尺寸
        dpi: 图表DPI
    """
    # 计算统计信息
    stats_dict = calculate_statistics(errors)
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 主Y轴：频率（直方图）
    ax1.set_xlabel('Error', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    
    # 绘制直方图
    if n_bins is None:
        n_bins = max(20, int(np.sqrt(len(errors))))  # 自动确定bins数量
    
    print(f"使用bins数量: {n_bins}")
    
    n, bins, patches = ax1.hist(errors, bins=n_bins, 
                                 density=False,  # 使用频率而不是密度
                                 alpha=0.7,
                                 color='purple',
                                 edgecolor='black',
                                 linewidth=0.5,
                                 label='Bars')
    
    # 创建第二个Y轴：概率密度
    ax2 = ax1.twinx()
    ax2.set_ylabel('Probability Density', fontsize=12)
    
    # 绘制正态分布拟合曲线（Bell Curve）
    x_range = np.linspace(errors.min(), errors.max(), 200)
    mu, sigma = stats_dict['mean'], stats_dict['std']
    bell_curve = norm.pdf(x_range, mu, sigma)
    ax2.plot(x_range, bell_curve, 'r-', linewidth=2, label='Bell Curve')
    
    # 绘制KDE曲线（核密度估计）
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(errors)
    kde_curve = kde(x_range)
    ax2.plot(x_range, kde_curve, 'b--', linewidth=2, label='KDE')
    
    # 添加统计信息框
    stats_text = f"Mean:     {stats_dict['mean']:>7.2f}\n"
    stats_text += f"Median:   {stats_dict['median']:>7.2f}\n"
    stats_text += f"Std Dev:  {stats_dict['std']:>7.2f}\n"
    stats_text += f"Skewness: {stats_dict['skewness']:>7.2f}\n"
    stats_text += f"Kurtosis: {stats_dict['kurtosis']:>7.2f}\n"
    stats_text += f"Range:    {stats_dict['range']:>7.2f}"
    
    # 在左上角添加统计信息框
    ax1.text(0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # 设置网格
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 设置标题
    ax1.set_title(title, fontsize=13, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n[保存] 误差分布图: {output_path}")
    print(f"\n统计摘要:")
    print(f"  样本数: {len(errors)}")
    print(f"  均值: {stats_dict['mean']:.2f}")
    print(f"  中位数: {stats_dict['median']:.2f}")
    print(f"  标准差: {stats_dict['std']:.2f}")
    print(f"  偏度: {stats_dict['skewness']:.2f}")
    print(f"  峰度: {stats_dict['kurtosis']:.2f}")
    print(f"  范围: [{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]")
    
    return stats_dict


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='绘制预测误差分布图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从单次预测结果生成
  python plot_error_distribution.py experiments/runs/.../predictions_detail/predictions_detail.csv
  
  # 从批量预测结果生成
  python plot_error_distribution.py experiments/runs/.../batch_engines/train/batch_summary.csv
  
  # 自定义输出路径和细粒度
  python plot_error_distribution.py data.csv --output error_dist.png --bins 50
  
  # 高细粒度（更多bins）
  python plot_error_distribution.py data.csv --bins 100
        """
    )
    
    parser.add_argument('csv_file', type=str,
                       help='CSV文件路径（包含Error列）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片路径（默认：与CSV同目录）')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图表DPI（默认：300）')
    parser.add_argument('--figsize', type=str, default='6,5',
                       help='图表大小 "宽,高"（默认：6,5）')
    parser.add_argument('--title', type=str, default='Prediction Error Distribution',
                       help='图表标题')
    parser.add_argument('--bins', type=int, default=None,
                       help='直方图bins数量（默认：自动计算）')
    
    args = parser.parse_args()
    
    # 读取CSV文件
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"[错误] CSV文件不存在: {csv_path}")
        sys.exit(1)
    
    print(f"\n加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 检查是否包含Error列
    if 'Error' not in df.columns:
        print(f"[错误] CSV文件必须包含 'Error' 列")
        print(f"可用列: {list(df.columns)}")
        sys.exit(1)
    
    # 获取误差数据
    errors = df['Error'].values
    print(f"误差样本数: {len(errors)}")
    
    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path.parent / 'error_distribution.png'
    
    # 解析图表大小
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
        if len(figsize) != 2:
            raise ValueError
    except:
        print(f"[警告] 无效的figsize格式，使用默认值 (6, 5)")
        figsize = (6, 5)
    
    # 绘制误差分布图
    stats_dict = plot_error_distribution(
        errors, output_path,
        title=args.title,
        figsize=figsize,
        dpi=args.dpi,
        n_bins=args.bins
    )
    
    print(f"\n✓ 完成！")


if __name__ == '__main__':
    main()

