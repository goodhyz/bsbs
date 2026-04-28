"""
第三章 Figure 3-2 / 3-3 / 3-4 生成脚本
计算并绘制：
  - Figure 3-2：学习效率（Learning Efficiency）柱状图
  - Figure 3-3：Anti-NFL 指标柱状图
  - Figure 3-4：LE vs. Anti-NFL 散点图

学习效率需要：
  1. test/bbob-10D/metadata/            （checkpoint-20 结果，ch3_test.py 阶段1产出）
  2. test/checkpoint-0/bbob-10D/metadata/ （checkpoint-0 结果，ch3_test.py 阶段2产出）
  3. train/{method}/checkpoint_log.txt   （训练过程时间日志）
若上述文件缺失，学习效率将跳过，仅输出 Anti-NFL 图。

Anti-NFL 只依赖 test/{problem}/metadata/（均为最终模型结果）。

运行目录：MetaBox 项目根目录
    python for_review/ch3_le_antinfl.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

METABBO_LIST = [
    'RNNOPT', 'DEDDQN', 'DEDQN', 'LDE',
    'RLPSO', 'RLEPSO', 'NRLPSO', 'LES',
    'GLEET', 'RLDAS', 'SYMBOL', 'RLDEAFL',
]
NO_ZERO_SHOT = {'RNNOPT', 'RLPSO', 'RLDAS'}
BBOB_PROBLEMS = {'bbob-10D', 'bbob-noisy-10D', 'bbob-30D', 'bbob-noisy-30D'}

ALL_PROBLEMS = [
    'bbob-10D', 'bbob-noisy-10D', 'bbob-30D',
    'bbob-noisy-30D', 'protein', 'uav', 'hpo-b',
]

OUTPUT_FILE = "figure_ch3_le_antinfl.pdf"


# ─────────────────────── 工具函数 ───────────────────────

def load_mean_performance(test_dir: str, method: str, gbest: dict) -> float:
    """计算某方法在给定测试目录下所有实例上的归一化平均性能."""
    meta_dir = os.path.join(test_dir, method)
    if not os.path.isdir(meta_dir):
        return None
    problem_files = [f for f in os.listdir(meta_dir) if f.endswith('.pkl')]
    perfs = []
    for fname in problem_files:
        p = fname.split('.')[0]
        with open(os.path.join(meta_dir, fname), 'rb') as f:
            metadata = pickle.load(f)
        for run in metadata:
            cost_seq = run['Cost']
            y_0  = float(np.min(cost_seq[0]))
            y_min = y_0
            for cost in cost_seq:
                y_min = min(y_min, float(np.min(cost)))
            g = (y_min - y_0) / (gbest.get(p, 0.0) - y_0 + 1e-20)
            perfs.append(g)
    return float(np.mean(perfs)) if perfs else None


def find_gbest(test_dir: str, problem_list: list, methods: list, is_bbob: bool) -> dict:
    gbest = {p: (0.0 if is_bbob else 1e32) for p in problem_list}
    if is_bbob:
        return gbest
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
                    y_min = min(y_min, float(np.min(cost)))
                gbest[p] = min(gbest[p], y_min)
    return gbest


def read_training_time(method: str) -> float | None:
    """从 train/{method}/checkpoint_log.txt 读取最终训练时间（秒）."""
    log_path = os.path.join('train', method, 'checkpoint_log.txt')
    if not os.path.exists(log_path):
        return None
    last_time = None
    with open(log_path, 'r') as f:
        for line in f:
            if 'Time:' in line:
                try:
                    time_str = line.strip().split('Time:')[1].split('s')[0].strip()
                    last_time = float(time_str)
                except (IndexError, ValueError):
                    pass
    return last_time


# ─────────────────────── Anti-NFL 计算 ───────────────────────

def compute_anti_nfl(perf_per_problem: dict) -> dict:
    """
    Anti-NFL = exp( mean_b [ (Perf_OOD_b - Perf_train) / Perf_train ] )
    perf_per_problem: {problem: {method: perf_value}}
    """
    train_key = 'bbob-10D'
    ood_keys  = [p for p in ALL_PROBLEMS if p != train_key]
    anti_nfl  = {}
    for method in METABBO_LIST:
        if perf_per_problem[train_key].get(method) is None:
            continue
        p_train = perf_per_problem[train_key][method]
        ratios  = []
        for ood in ood_keys:
            p_ood = perf_per_problem[ood].get(method)
            if p_ood is not None:
                ratios.append((p_ood - p_train) / (p_train + 1e-20))
        if ratios:
            anti_nfl[method] = float(np.exp(np.mean(ratios)))
    return anti_nfl


# ─────────────────────── 主流程 ───────────────────────

if __name__ == "__main__":
    # ── 1. 收集所有问题集上的方法性能 ──
    perf_per_problem = {}
    for problem in ALL_PROBLEMS:
        test_dir  = f"test/{problem}/metadata"
        is_bbob   = problem in BBOB_PROBLEMS
        is_10D    = "10D" in problem
        active    = [m for m in METABBO_LIST if is_10D or m not in NO_ZERO_SHOT]

        # 获取问题实例列表
        ref = next((m for m in active if os.path.isdir(os.path.join(test_dir, m))), None)
        if ref is None:
            perf_per_problem[problem] = {m: None for m in METABBO_LIST}
            print(f"  [WARN] {problem} 测试数据缺失，请先运行 ch3_test.py")
            continue

        prob_files = [f.split('.')[0] for f in os.listdir(os.path.join(test_dir, ref))
                      if f.endswith('.pkl')]
        gbest = find_gbest(test_dir, prob_files, active, is_bbob)

        perf_per_problem[problem] = {}
        for method in METABBO_LIST:
            if not is_10D and method in NO_ZERO_SHOT:
                perf_per_problem[problem][method] = None
                continue
            perf_per_problem[problem][method] = load_mean_performance(
                test_dir, method, gbest
            )

    # ── 2. Anti-NFL ──
    anti_nfl = compute_anti_nfl(perf_per_problem)
    print("Anti-NFL 结果：")
    for m, v in anti_nfl.items():
        print(f"  {m:12s}: {v:.4f}")

    # ── 3. 学习效率（可选） ──
    compute_le = True
    Tg = {}
    for method in METABBO_LIST:
        t = read_training_time(method)
        if t is None:
            print(f"  [WARN] 未找到 {method} 的训练日志，跳过学习效率计算。")
            compute_le = False
            break
        Tg[method] = t

    le = {}
    if compute_le:
        c0_dir  = "test/checkpoint-0/bbob-10D/metadata"
        c20_dir = "test/bbob-10D/metadata"
        ref = next((m for m in METABBO_LIST if os.path.isdir(os.path.join(c0_dir, m))), None)
        if ref is None:
            print("[WARN] checkpoint-0 测试数据缺失，跳过学习效率计算。")
            compute_le = False

    if compute_le:
        # 对 bbob-10D 统一使用 0 作为最优（BBOB 标准最优）
        prob_files = [f.split('.')[0] for f in os.listdir(os.path.join(c20_dir, METABBO_LIST[0]))
                      if f.endswith('.pkl')]
        gbest_10D = {p: 0.0 for p in prob_files}

        for method in METABBO_LIST:
            perf_c0  = load_mean_performance(c0_dir,  method, gbest_10D)
            perf_c20 = load_mean_performance(c20_dir, method, gbest_10D)
            if perf_c0 is None or perf_c20 is None:
                le[method] = None
                continue
            le[method] = (perf_c20 - perf_c0) / (Tg[method] + 1e-20)

        print("\n学习效率结果：")
        for m, v in le.items():
            print(f"  {m:12s}: {v:.6f}" if v is not None else f"  {m:12s}: N/A")

    # ── 4. 绘图 ──
    valid_methods = [m for m in METABBO_LIST if m in anti_nfl]
    anti_nfl_vals = [anti_nfl[m] for m in valid_methods]

    if compute_le:
        le_vals = [le.get(m) for m in valid_methods]
        valid_le = [(m, l, a) for m, l, a in zip(valid_methods, le_vals, anti_nfl_vals)
                    if l is not None]

    n_plots = 3 if compute_le else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Figure 3-3：Anti-NFL 柱状图
    ax_antinfl = axes[-1] if compute_le else axes[0]
    ax_antinfl.bar(valid_methods, anti_nfl_vals, color='salmon', edgecolor='white')
    ax_antinfl.set_ylabel('Anti-NFL')
    ax_antinfl.set_title('Figure 3-3: Anti-NFL')
    ax_antinfl.tick_params(axis='x', rotation=45)
    ax_antinfl.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='baseline=1')
    ax_antinfl.legend(fontsize=8)

    if compute_le:
        le_methods  = [t[0] for t in valid_le]
        le_vals_plt = [t[1] for t in valid_le]
        an_vals_plt = [t[2] for t in valid_le]

        # Figure 3-2：LE 柱状图
        axes[0].bar(le_methods, le_vals_plt, color='skyblue', edgecolor='white')
        axes[0].set_ylabel('Learning Efficiency (perf / s)')
        axes[0].set_title('Figure 3-2: Learning Efficiency')
        axes[0].tick_params(axis='x', rotation=45)

        # Figure 3-4：LE vs Anti-NFL 散点图
        axes[1].scatter(le_vals_plt, an_vals_plt, color='green', zorder=5)
        for m, le_v, an_v in zip(le_methods, le_vals_plt, an_vals_plt):
            axes[1].annotate(m, (le_v, an_v), fontsize=7,
                             xytext=(4, 2), textcoords='offset points')
        axes[1].set_xlabel('Learning Efficiency')
        axes[1].set_ylabel('Anti-NFL')
        axes[1].set_title('Figure 3-4: LE vs. Anti-NFL')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_FILE.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"\nFigure 3-2/3-3/3-4 已保存至 {OUTPUT_FILE}")
    plt.show()
