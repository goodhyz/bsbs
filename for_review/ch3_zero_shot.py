"""
第三章 Figure 3-1 生成脚本
绘制 6 个分布外问题集上的零样本泛化曲线（归一化性能 vs. FEs）。
每个子图代表一个 OOD 问题集，曲线按方法着色，带标准差阴影。

输出：figure_ch3_zero_shot.pdf / .png

运行目录：MetaBox 项目根目录
    python for_review/ch3_zero_shot.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

METABBO_LIST = [
    'RNNOPT', 'DEDDQN', 'DEDQN', 'LDE',
    'RLPSO', 'RLEPSO', 'NRLPSO', 'LES',
    'GLEET', 'RLDAS', 'SYMBOL', 'RLDEAFL',
]
BBO_LIST = ['PSO', 'DE', 'SHADE', 'JDE21', 'MADDE']
BASELINE_LIST = METABBO_LIST + BBO_LIST

# 不支持零样本迁移的方法（仅限 10D 问题）
NO_ZERO_SHOT = {'RNNOPT', 'RLPSO', 'RLDAS'}

# BBOB 类问题全局最优已知为 0
BBOB_PROBLEMS = {'bbob-noisy-10D', 'bbob-30D', 'bbob-noisy-30D'}

OOD_PROBLEMS = [
    ('bbob-noisy-10D', 'Noisy-10D'),
    ('bbob-30D',       '30D'),
    ('bbob-noisy-30D', 'Noisy-30D'),
    ('protein',        'Protein'),
    ('uav',            'UAV'),
    ('hpo-b',          'HPO-B'),
]

OUTPUT_PDF = "figure_ch3_zero_shot.pdf"
OUTPUT_PNG = "figure_ch3_zero_shot.png"


def load_problem_list(test_dir: str, ref_method: str) -> list[str]:
    ref_path = os.path.join(test_dir, ref_method)
    return sorted(f.split('.')[0] for f in os.listdir(ref_path) if f.endswith('.pkl'))


def find_global_best(test_dir: str, problem_list: list, methods: list, is_bbob: bool) -> dict:
    """找每个实例的全局最优（合并所有方法的最优值）."""
    gbest = {}
    for p in problem_list:
        gbest[p] = 0.0 if is_bbob else 1e32
    if is_bbob:
        return gbest  # BBOB 最优固定为 0
    for method in methods:
        for p in problem_list:
            path = os.path.join(test_dir, method, f"{p}.pkl")
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                metadata = pickle.load(f)
            for run in metadata:
                y_min = np.min(run['Cost'][0])
                for cost in run['Cost']:
                    y_min = min(y_min, np.min(cost))
                gbest[p] = min(gbest[p], y_min)
    return gbest


def compute_zero_shot_curve(test_dir: str, method: str, problem_list: list, gbest: dict) -> dict:
    """计算某方法在某问题集上的平均性能曲线."""
    fes_ref = None       # 以最长轨迹为参照建立 FES 坐标轴
    perf_per_problem = []

    for p in problem_list:
        path = os.path.join(test_dir, method, f"{p}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            metadata = pickle.load(f)

        # 建立 FES 坐标（只做一次，取第一个问题的最长 run）
        if fes_ref is None:
            max_len = max(len(run['Cost']) for run in metadata)
            ref_run = next(r for r in metadata if len(r['Cost']) == max_len)
            fes_ref = []
            cumulative = 0
            for cost in ref_run['Cost']:
                cumulative += len(cost)
                fes_ref.append(cumulative)

        T = len(fes_ref)
        runs_perf = []
        for run in metadata:
            cost_seq = run['Cost']
            y_0  = float(np.min(cost_seq[0]))
            y_min = y_0
            traj = []
            for cost in cost_seq:
                y_min = min(y_min, float(np.min(cost)))
                g = (y_min - y_0) / (gbest[p] - y_0 + 1e-20)
                traj.append(g)
            # 对齐到 T 步（短轨迹末尾值填充）
            while len(traj) < T:
                traj.append(traj[-1])
            runs_perf.append(traj[:T])

        perf_per_problem.append(np.mean(runs_perf, axis=0))  # shape: (T,)

    mean_curve = np.mean(perf_per_problem, axis=0)
    std_curve  = np.std(perf_per_problem, axis=0)
    return {'fes': np.array(fes_ref), 'mean': mean_curve, 'std': std_curve}


if __name__ == "__main__":
    # 配色：MetaBBO 用 tab20，BBO 用灰色系
    colors  = list(cm.tab20.colors[:len(METABBO_LIST)])
    bbo_colors = ['#888', '#aaa', '#bbb', '#ccc', '#ddd']
    method_colors = {m: c for m, c in zip(METABBO_LIST, colors)}
    for m, c in zip(BBO_LIST, bbo_colors):
        method_colors[m] = c

    linestyles = {m: '-'  for m in METABBO_LIST}
    linestyles.update({m: '--' for m in BBO_LIST})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax_idx, (problem, label) in enumerate(OOD_PROBLEMS):
        ax = axes[ax_idx]
        test_dir = f"test/{problem}/metadata"
        is_bbob = problem in BBOB_PROBLEMS
        is_10D  = "10D" in problem

        # 确定参与本问题集的方法
        active_methods = [m for m in BASELINE_LIST if is_10D or m not in NO_ZERO_SHOT]
        ref_method = next(m for m in active_methods if m in METABBO_LIST)

        if not os.path.isdir(os.path.join(test_dir, ref_method)):
            ax.text(0.5, 0.5, f'数据缺失\n请先运行 ch3_test.py',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        problem_list = load_problem_list(test_dir, ref_method)
        gbest = find_global_best(test_dir, problem_list, active_methods, is_bbob)

        for method in active_methods:
            result = compute_zero_shot_curve(test_dir, method, problem_list, gbest)
            if result is None:
                continue
            fes, mean, std = result['fes'], result['mean'], result['std']
            ax.plot(fes, mean,
                    label=method,
                    color=method_colors[method],
                    linestyle=linestyles[method],
                    linewidth=1.5 if method in METABBO_LIST else 1.0)
            ax.fill_between(fes, mean - std, mean + std,
                            alpha=0.15, color=method_colors[method])

        ax.set_title(label, fontsize=12)
        ax.set_xlabel('FEs', fontsize=10)
        ax.set_ylabel('Normalized Performance', fontsize=10)
        ax.grid(True, alpha=0.3)

    # 统一图例放在最后一格或底部
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle('Figure 3-1: Zero-Shot Generalization on OOD Problems', fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Figure 3-1 已保存至 {OUTPUT_PDF} / {OUTPUT_PNG}")
    plt.show()
