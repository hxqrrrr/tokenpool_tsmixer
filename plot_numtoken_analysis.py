"""
Token数量分析：绘制Token数量与模型性能的关系图（论文风格）
Analyze the impact of number of tokens on model performance (Publication Style)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# 设置论文风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6
plt.rcParams['pdf.fonttype'] = 42  # TrueType字体（可编辑）
plt.rcParams['ps.fonttype'] = 42


def plot_numtoken_vs_score(csv_path, output_dir='experiments/analysis', dpi=300):
    """
    绘制Token数量 vs Score 的关系图
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
        dpi: 图片分辨率
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 找到最优Token数量
    best_idx = df['Score'].idxmin()
    best_numtoken = df.loc[best_idx, 'Num_Tokens']
    best_score = df.loc[best_idx, 'Score']
    best_rmse = df.loc[best_idx, 'RMSE']
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建图表 - 论文风格
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    
    # 学术配色方案（适合黑白打印）
    color_score = '#2C3E50'  # 深灰蓝
    color_rmse = '#E74C3C'   # 深红
    color_best = '#27AE60'   # 深绿
    
    # ========== 子图1: Num Tokens vs Score ==========
    line1 = ax1.plot(df['Num_Tokens'], df['Score'], 
                     marker='o', linewidth=1.8, markersize=5,
                     color=color_score, label='Score', 
                     markerfacecolor='white', markeredgewidth=1.5)
    
    # 标记最优点
    ax1.plot(best_numtoken, best_score, 
             marker='s', markersize=8, 
             color=color_best, 
             markeredgecolor='black', markeredgewidth=1,
             label=f'Optimal (N={int(best_numtoken)})',
             zorder=5)
    
    # 简洁的标注
    ax1.annotate(f'({int(best_numtoken)}, {best_score:.1f})',
                xy=(best_numtoken, best_score),
                xytext=(15, 20), textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor=color_best, 
                         linewidth=1.5),
                arrowprops=dict(arrowstyle='->', 
                              color=color_best, 
                              lw=1.2))
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('(a) Number of Tokens vs. Score', 
                  fontsize=11, loc='left', pad=10)
    ax1.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='gray')
    
    # ========== 子图2: Num Tokens vs RMSE ==========
    best_rmse_idx = df['RMSE'].idxmin()
    best_rmse_numtoken = df.loc[best_rmse_idx, 'Num_Tokens']
    best_rmse_value = df.loc[best_rmse_idx, 'RMSE']
    
    line2 = ax2.plot(df['Num_Tokens'], df['RMSE'], 
                     marker='D', linewidth=1.8, markersize=4.5,
                     color=color_rmse, label='RMSE',
                     markerfacecolor='white', markeredgewidth=1.5)
    
    # 标记最优点
    ax2.plot(best_rmse_numtoken, best_rmse_value, 
             marker='s', markersize=8, 
             color=color_best,
             markeredgecolor='black', markeredgewidth=1,
             label=f'Optimal (N={int(best_rmse_numtoken)})',
             zorder=5)
    
    # 简洁的标注
    ax2.annotate(f'({int(best_rmse_numtoken)}, {best_rmse_value:.2f})',
                xy=(best_rmse_numtoken, best_rmse_value),
                xytext=(15, 20), textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor=color_best, 
                         linewidth=1.5),
                arrowprops=dict(arrowstyle='->', 
                              color=color_best, 
                              lw=1.2))
    
    ax2.set_xlabel('Number of Tokens', fontsize=11)
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('(b) Number of Tokens vs. RMSE', 
                  fontsize=11, loc='left', pad=10)
    ax2.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='gray')
    
    # 设置x轴为对数刻度（因为范围很大：1-300）
    ax2.set_xscale('log')
    ax1.set_xscale('log')
    
    # 设置主要刻度
    major_ticks = [1, 2, 4, 6, 10, 20, 40, 60, 100, 200, 300]
    ax2.set_xticks(major_ticks)
    ax2.set_xticklabels([str(x) for x in major_ticks])
    
    plt.tight_layout()
    
    # 保存图片（PNG + PDF）
    output_path_png = output_dir / 'numtoken_analysis.png'
    output_path_pdf = output_dir / 'numtoken_analysis.pdf'
    plt.savefig(output_path_png, dpi=dpi, bbox_inches='tight')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path_png}")
    print(f"✓ PDF已保存: {output_path_pdf}")
    
    plt.close()
    
    # ========== 创建单独的Score图（论文风格） ==========
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # 主曲线
    ax.plot(df['Num_Tokens'], df['Score'], 
            marker='o', linewidth=2, markersize=5,
            color=color_score, 
            markerfacecolor='white', 
            markeredgewidth=1.5,
            label='Test Score', 
            zorder=3)
    
    # 淡化填充区域
    ax.fill_between(df['Num_Tokens'], df['Score'], 
                     alpha=0.08, color=color_score)
    
    # 标记最优点
    ax.plot(best_numtoken, best_score, 
            marker='s', markersize=10, 
            color=color_best,
            markeredgecolor='black',
            markeredgewidth=1.5,
            label=f'Optimal: N={int(best_numtoken)}',
            zorder=5)
    
    # 添加最优点标注
    ax.annotate(f'Min: {best_score:.1f}',
               xy=(best_numtoken, best_score),
               xytext=(20, 25), textcoords='offset points',
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', 
                        facecolor='white', 
                        edgecolor=color_best, 
                        linewidth=1.5),
               arrowprops=dict(arrowstyle='->', 
                             color=color_best, 
                             lw=1.5))
    
    # 添加参考线
    ax.axhline(y=best_score, color=color_best, linestyle=':', 
              alpha=0.4, linewidth=1.2)
    ax.axvline(x=best_numtoken, color=color_best, linestyle=':', 
              alpha=0.4, linewidth=1.2)
    
    ax.set_xlabel('Number of Tokens', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Impact of Token Number on Model Performance', 
                fontsize=12, pad=12)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    # 对数刻度
    ax.set_xscale('log')
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(x) for x in major_ticks])
    
    plt.tight_layout(pad=0.5)
    
    # 保存单独的Score图
    score_output_png = output_dir / 'numtoken_score_plot.png'
    score_output_pdf = output_dir / 'numtoken_score_plot.pdf'
    plt.savefig(score_output_png, dpi=dpi, bbox_inches='tight')
    plt.savefig(score_output_pdf, format='pdf', bbox_inches='tight')
    print(f"✓ Score图已保存: {score_output_png}")
    print(f"✓ PDF已保存: {score_output_pdf}")
    
    plt.close()
    
    # ========== 打印统计信息 ==========
    print(f"\n{'='*60}")
    print(f"Token数量分析结果")
    print(f"{'='*60}")
    print(f"总实验次数: {len(df)}")
    print(f"Token数量范围: {df['Num_Tokens'].min():.0f} - {df['Num_Tokens'].max():.0f}")
    print(f"\n最佳Score:")
    print(f"  Token数量: {int(best_numtoken)}")
    print(f"  Score: {best_score:.4f}")
    print(f"  RMSE: {best_rmse:.4f}")
    print(f"\n最佳RMSE:")
    print(f"  Token数量: {int(best_rmse_numtoken)}")
    print(f"  RMSE: {best_rmse_value:.4f}")
    print(f"  Score: {df.loc[best_rmse_idx, 'Score']:.4f}")
    print(f"\nScore统计:")
    print(f"  平均值: {df['Score'].mean():.2f}")
    print(f"  中位数: {df['Score'].median():.2f}")
    print(f"  标准差: {df['Score'].std():.2f}")
    print(f"  最大值: {df['Score'].max():.2f} (Tokens={int(df.loc[df['Score'].idxmax(), 'Num_Tokens'])})")
    print(f"  最小值: {df['Score'].min():.2f} (Tokens={int(df.loc[df['Score'].idxmin(), 'Num_Tokens'])})")
    print(f"\nRMSE统计:")
    print(f"  平均值: {df['RMSE'].mean():.4f}")
    print(f"  中位数: {df['RMSE'].median():.4f}")
    print(f"  标准差: {df['RMSE'].std():.4f}")
    print(f"  最大值: {df['RMSE'].max():.4f} (Tokens={int(df.loc[df['RMSE'].idxmax(), 'Num_Tokens'])})")
    print(f"  最小值: {df['RMSE'].min():.4f} (Tokens={int(df.loc[df['RMSE'].idxmin(), 'Num_Tokens'])})")
    print(f"{'='*60}\n")
    
    # 趋势分析
    print(f"{'='*60}")
    print(f"趋势分析")
    print(f"{'='*60}")
    
    # 计算相关性（对数尺度）
    correlation = np.log(df['Num_Tokens']).corr(df['Score'])
    print(f"Token数量(log)与Score的相关性: {correlation:.4f}")
    
    # 寻找最优区间
    optimal_score_threshold = best_score * 1.05  # 允许5%的性能损失
    optimal_tokens = df[df['Score'] <= optimal_score_threshold]['Num_Tokens']
    if len(optimal_tokens) > 0:
        print(f"\n推荐Token数量范围 (Score ≤ {optimal_score_threshold:.2f}):")
        print(f"  {int(optimal_tokens.min())} - {int(optimal_tokens.max())}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='绘制Token数量分析图'
    )
    parser.add_argument('csv_file', type=str, nargs='?',
                       default='experiments/analysis/numtoken_results.csv',
                       help='CSV数据文件路径')
    parser.add_argument('--output-dir', type=str,
                       default='experiments/analysis',
                       help='输出目录')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图片分辨率')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.csv_file).exists():
        print(f"错误: 找不到文件 {args.csv_file}")
        return
    
    # 绘图
    plot_numtoken_vs_score(args.csv_file, args.output_dir, args.dpi)


if __name__ == '__main__':
    main()

