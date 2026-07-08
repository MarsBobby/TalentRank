import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置样式（匹配你偏好的现代、清晰布局）
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False

def visualize_ahp_convergence(matrix, feature_names, title="AHP 主观矩阵与一致性审计"):
    n = matrix.shape[0]
    
    # 1. 计算最大特征值与特征向量
    eig_vals, eig_vecs = np.linalg.eig(matrix)
    max_idx = np.argmax(eig_vals.real)
    lambda_max = eig_vals[max_idx].real
    
    # 提取对应特征向量并归一化得到权重
    w_ahp = eig_vecs[:, max_idx].real
    w_ahp = w_ahp / np.sum(w_ahp)
    
    # 2. 一致性校验 (以 8 维特征为例，RI = 1.41)
    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.89, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    ri = RI_dict.get(n, 1.41)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    cr = ci / ri if ri > 0 else 0
    
    # 3. 开始绘图 (画布：左侧矩阵热力图，右侧权重及CR看板)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # 左图：专家判断矩阵对数热力图（便于观察 1/9 到 9 的跨度）
    sns.heatmap(
        np.log2(matrix), 
        annot=matrix, 
        fmt=".2f", 
        cmap="RdYlBu_r", 
        cbar=False,
        xticklabels=feature_names, 
        yticklabels=feature_names,
        ax=axes[0],
        annot_kws={"size": 10, "weight": "bold"}
    )
    axes[0].set_title(f"{title} - 专家成对比较矩阵 $\mathbf{{A}}$", fontsize=13, pad=15)
    
    # 右图：主观权重条形图
    y_pos = np.arange(len(feature_names))
    bars = axes[1].barh(y_pos, w_ahp, color="#4A90E2", edgecolor="none", height=0.6)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(feature_names, fontsize=11)
    axes[1].invert_yaxis()  # 让顶层特征排在上面
    axes[1].set_xlabel("主观权重值 $\mathbf{{W}}_{AHP}$", fontsize=11)
    
    # 在条形图上标注具体权重值
    for bar in bars:
        width = bar.get_width()
        axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                     va='center', ha='left', fontsize=10, weight='bold', color='#333333')
        
    # 4. 强制注入一致性熔断指示器
    status_color = "#2ECC71" if cr < 0.1 else "#E74C3C"
    status_text = f"CR = {cr:.4f} < 0.1 (通过校验 · 准予沉淀)" if cr < 0.1 else f"CR = {cr:.4f} \geq 0.1 (熔断驳回 · 重新校准)"
    
    # 在右图上方放置审计状态标签
    axes[1].text(0.5, 1.08, status_text, transform=axes[1].transAxes,
                 ha="center", va="center", color="white", weight="bold", fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, edgecolor="none"))
    
    axes[1].set_title(f"$\lambda_{{max}} = {lambda_max:.3f}$ | $CI = {ci:.3f}$", fontsize=11, pad=35, color="#666666")
    
    plt.tight_layout()
    plt.show()

# ── 测试运行 ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 模拟一个符合一致性的 8x8 品牌领域专家矩阵
    features = ["学历评分", "职级标定", "晋升斜率", "管理跨度", "KPI绩效", "稳定性", "成就信号", "软实力"]
    
    # 构造一个基本调和的矩阵 (CR < 0.1)
    base_weights = np.array([0.05, 0.15, 0.10, 0.08, 0.35, 0.07, 0.15, 0.05])
    mock_matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            mock_matrix[i, j] = base_weights[i] / base_weights[j]
            
    # 稍微人为加入一点专家噪声，测试收敛审计能力
    mock_matrix[4, 1] *= 1.1
    mock_matrix[1, 4] = 1.0 / mock_matrix[4, 1]
    
    visualize_ahp_convergence(mock_matrix, features, title="品牌职能 (Branding)")