"""
第四章分析脚本（Table 4-1 / Figure 4-1 / Table 4-2）
从 test/ 目录读取结果（由 ch3_test.py + ch4_test.py 产出），针对第四章
选定的对比方法进行分析。

生成内容：
  - Table 4-1：bbob-10D 上各方法的 mean/std/rank（Excel）
  - Figure 4-1：6 个 OOD 问题集的零样本泛化曲线
  - Table 4-2：消融实验结果（RLDEAFL_T vs RLDEAFL vs GLEET，Excel）

运行目录：MetaBox 项目根目录
    python for_review/ch4_analysis.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── 第四章方法列表 ──
CH4_METABBO = ['RLDEAFL_T', 'RLDEAFL', 'GLEET', 'RLDAS', 'LDE']
CH4_BBO     = ['DE', 'SHADE', 'JDE21', 'MADDE']
CH4_ALL     = CH4_METABBO + CH4_BBO

# 消融组（Table 4-2）：仅关注三者
ABLATION_LIST = ['RLDEAFL_T', 'RLDEAFL', 'GLEET']

NO_ZERO_SHOT = {'RLDAS'}

OOD_PROBLEMS = [
    ('bbob-noisy-10D', 'Noisy-10D'),
    ('bbob-30D',       '30D'),
    ('bbob-noisy-30D', 'Noisy-30D'),
    ('protein',        'Protein'),
    ('uav',            'UAV'),
    ('hpo-b',          'HPO-B'),
]
BBOB_PROBLEMS = {'bbob-noisy-10D', 'bbob-30D', 'bbob-noisy-30D'}

OUTPUT_TABLE = "result_ch4_table.xlsx"
OUTPUT_FIGURE = "figure_ch4_zero_shot.pdf"


# ─────────────────────── 公共工具 ───────────────────────

def load_final_cost(pkl_path: str) -> tuple[float, float]:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    finals = []
    for run in data:
        y_min = np.min(run['Cost'][0])
        for cost in run['Cost']:
            y_min = min(y_min, float(np.min(cost)))
        finals.append(y_min)
    return float(np.mean(finals)), float(np.std(finals))


def compute_ranks(mean_mat: np.ndarray, std_mat: np.ndarray) -> np.ndarray:
    m, n = mean_mat.shape
    ranks = np.zeros((m, n))
    for col in range(n):
        order = sorted(range(m),
                       key=lambda i: (mean_mat[i, col], std_mat[i, col]))
        rank, i = 1, 0
        while i < m:
            j = i
            while j + 1 < m and (
                mean_mat[order[j], col] == mean_mat[order[j + 1], col] and
                std_mat[order[j], col] == std_mat[order[j + 1], col]
            ):
                j += 1
            for k in range(i, j + 1):
                ranks[order[k], col] = rank
            rank += (j - i + 1)
            i = j + 1
    return ranks


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


def compute_zero_shot_curve(test_dir: str, method: str, problem_list: list, gbest: dict):
    perf_per_problem = []
    fes_ref = None
    for p in problem_list:
        path = os.path.join(test_dir, method, f"{p}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            metadata = pickle.load(f)
        if fes_ref is None:
            max_len = max(len(run['Cost']) for run in metadata)
            ref_run = next(r for r in metadata if len(r['Cost']) == max_len)
            cumulative, fes_ref = 0, []
            for cost in ref_run['Cost']:
                cumulative += len(cost)
                fes_ref.append(cumulative)
        T = len(fes_ref)
        runs = []
        for run in metadata:
            y_0 = float(np.min(run['Cost'][0]))
            y_min = y_0
            traj = []
            for cost in run['Cost']:
                y_min = min(y_min, float(np.min(cost)))
                traj.append((y_min - y_0) / (gbest[p] - y_0 + 1e-20))
            while len(traj) < T:
                traj.append(traj[-1])
            runs.append(traj[:T])
        perf_per_problem.append(np.mean(runs, axis=0))
    return {
        'fes':  np.array(fes_ref),
        'mean': np.mean(perf_per_problem, axis=0),
        'std':  np.std(perf_per_problem, axis=0),
    }


# ─────────────────────── Table 4-1 ───────────────────────

def make_table_4_1():
    test_dir = "test/bbob-10D/metadata"
    ref = next((m for m in CH4_ALL if os.path.isdir(os.path.join(test_dir, m))), None)
    if ref is None:
        print("[SKIP] Table 4-1：test/bbob-10D/metadata 数据缺失")
        return
    problem_list = sorted(f.split('.')[0] for f in os.listdir(os.path.join(test_dir, ref))
                          if f.endswith('.pkl'))
    m, n = len(CH4_ALL), len(problem_list)
    mean_mat = np.full((m, n), np.nan)
    std_mat  = np.full((m, n), np.nan)
    for i, method in enumerate(CH4_ALL):
        for j, problem in enumerate(problem_list):
            pkl = os.path.join(test_dir, method, f"{problem}.pkl")
            if os.path.exists(pkl):
                mean_mat[i, j], std_mat[i, j] = load_final_cost(pkl)
    rank_mat = compute_ranks(mean_mat, std_mat)
    avg_rank  = np.nanmean(rank_mat, axis=1)
    sorted_idx = np.argsort(avg_rank)
    final_rank = np.empty(m, dtype=int)
    final_rank[sorted_idx] = np.arange(1, m + 1)

    df_mean  = pd.DataFrame(mean_mat, index=CH4_ALL, columns=problem_list)
    df_std   = pd.DataFrame(std_mat,  index=CH4_ALL, columns=problem_list)
    df_rank  = pd.DataFrame(rank_mat, index=CH4_ALL, columns=problem_list)
    df_rank['avg_rank']   = avg_rank
    df_rank['final_rank'] = final_rank

    combined = {
        p: [f"{mean_mat[i, j]:.4e} ± {std_mat[i, j]:.2e}" for i in range(m)]
        for j, p in enumerate(problem_list)
    }
    df_comb = pd.DataFrame(combined, index=CH4_ALL)
    df_comb['avg_rank']   = avg_rank.round(2)
    df_comb['final_rank'] = final_rank

    with pd.ExcelWriter(OUTPUT_TABLE, engine='openpyxl') as writer:
        df_comb.to_excel(writer, sheet_name='Table4-1 (mean±std+rank)')
        df_mean.to_excel(writer, sheet_name='mean (raw)')
        df_std.to_excel(writer,  sheet_name='std (raw)')
        df_rank.to_excel(writer, sheet_name='rank')
    print(f"Table 4-1 已保存至 {OUTPUT_TABLE}")


# ─────────────────────── Table 4-2（消融） ───────────────────────

def make_table_4_2(writer):
    """将消融结果追加到同一 Excel 文件."""
    test_dir = "test/bbob-10D/metadata"
    ref = next((m for m in ABLATION_LIST if os.path.isdir(os.path.join(test_dir, m))), None)
    if ref is None:
        print("[SKIP] Table 4-2：消融数据缺失")
        return
    problem_list = sorted(f.split('.')[0] for f in os.listdir(os.path.join(test_dir, ref))
                          if f.endswith('.pkl'))
    m, n = len(ABLATION_LIST), len(problem_list)
    mean_mat = np.full((m, n), np.nan)
    std_mat  = np.full((m, n), np.nan)
    for i, method in enumerate(ABLATION_LIST):
        for j, problem in enumerate(problem_list):
            pkl = os.path.join(test_dir, method, f"{problem}.pkl")
            if os.path.exists(pkl):
                mean_mat[i, j], std_mat[i, j] = load_final_cost(pkl)
    rank_mat  = compute_ranks(mean_mat, std_mat)
    avg_rank  = np.nanmean(rank_mat, axis=1)

    df_abl = pd.DataFrame(
        {p: [f"{mean_mat[i, j]:.4e} ± {std_mat[i, j]:.2e}" for i in range(m)]
         for j, p in enumerate(problem_list)},
        index=ABLATION_LIST
    )
    df_abl['avg_rank'] = avg_rank.round(2)

    df_abl.to_excel(writer, sheet_name='Table4-2 (ablation)')
    print("Table 4-2 消融结果已追加至 Excel")


# ─────────────────────── Figure 4-1 ───────────────────────

def make_figure_4_1():
    colors = list(cm.tab10.colors)
    method_colors = {m: colors[i % len(colors)] for i, m in enumerate(CH4_ALL)}
    linestyles = {m: '-' for m in CH4_METABBO}
    linestyles.update({m: '--' for m in CH4_BBO})

    # 突出显示核心对比方法
    highlight = {'RLDEAFL_T': 2.5, 'RLDEAFL': 2.0, 'GLEET': 1.5}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax_idx, (problem, label) in enumerate(OOD_PROBLEMS):
        ax = axes[ax_idx]
        test_dir = f"test/{problem}/metadata"
        is_bbob  = problem in BBOB_PROBLEMS
        is_10D   = "10D" in problem
        active   = [m for m in CH4_ALL if is_10D or m not in NO_ZERO_SHOT]

        ref = next((m for m in active if os.path.isdir(os.path.join(test_dir, m))), None)
        if ref is None:
            ax.text(0.5, 0.5, '数据缺失', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        problem_list = sorted(f.split('.')[0] for f in os.listdir(os.path.join(test_dir, ref))
                              if f.endswith('.pkl'))
        gbest = find_gbest(test_dir, problem_list, active, is_bbob)

        for method in active:
            result = compute_zero_shot_curve(test_dir, method, problem_list, gbest)
            if result is None:
                continue
            lw = highlight.get(method, 1.0)
            ax.plot(result['fes'], result['mean'],
                    label=method, color=method_colors[method],
                    linestyle=linestyles.get(method, '-'),
                    linewidth=lw)
            ax.fill_between(result['fes'],
                            result['mean'] - result['std'],
                            result['mean'] + result['std'],
                            alpha=0.15, color=method_colors[method])

        ax.set_title(label, fontsize=12)
        ax.set_xlabel('FEs', fontsize=10)
        ax.set_ylabel('Normalized Performance', fontsize=10)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle('Figure 4-1: Zero-Shot Generalization (Chapter 4 Methods)', fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_FIGURE.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Figure 4-1 已保存至 {OUTPUT_FIGURE}")
    plt.show()


# ─────────────────────── 主流程 ───────────────────────

if __name__ == "__main__":
    make_table_4_1()

    # 追加消融表到同一 Excel
    if os.path.exists(OUTPUT_TABLE):
        with pd.ExcelWriter(OUTPUT_TABLE, engine='openpyxl', mode='a') as writer:
            make_table_4_2(writer)
    else:
        print("[WARN] 请先运行 make_table_4_1() 以创建 Excel 文件")

    make_figure_4_1()
