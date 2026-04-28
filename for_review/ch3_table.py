"""
第三章 Table 3-1 / Table 3-2 生成脚本
从 test/bbob-10D/metadata/ 读取测试结果，输出：
  - Table 3-1：各方法在每个 bbob-10D 实例上的 mean ± std
  - Table 3-2：各方法的逐实例排名、平均排名、最终排名
结果保存至 result_ch3_table.xlsx

运行目录：MetaBox 项目根目录
    python for_review/ch3_table.py
"""

import os
import numpy as np
import pandas as pd
import pickle

TEST_DIR = "test/bbob-10D/metadata"

METABBO_LIST = [
    'RNNOPT', 'DEDDQN', 'DEDQN', 'LDE',
    'RLPSO', 'RLEPSO', 'NRLPSO', 'LES',
    'GLEET', 'RLDAS', 'SYMBOL', 'RLDEAFL',
]
BBO_LIST = ['PSO', 'DE', 'SHADE', 'JDE21', 'MADDE']
BASELINE_LIST = METABBO_LIST + BBO_LIST

OUTPUT_FILE = "result_ch3_table.xlsx"


def load_final_cost(pkl_path: str) -> tuple[float, float]:
    """读取单个问题实例的测试结果，返回 (mean, std) 终值."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)  # List[run_dict]
    finals = []
    for run in data:
        cost_seq = run['Cost']       # List[array]，每代的函数值
        y_min = np.min(cost_seq[0])
        for cost in cost_seq:
            y_min = min(y_min, np.min(cost))
        finals.append(y_min)
    return float(np.mean(finals)), float(np.std(finals))


def compute_ranks(mean_mat: np.ndarray, std_mat: np.ndarray) -> np.ndarray:
    """按 (mean, std) 升序计算每列（问题实例）的方法排名，处理并列情况."""
    m, n = mean_mat.shape
    ranks = np.zeros((m, n))
    for col in range(n):
        order = sorted(
            range(m),
            key=lambda i: (mean_mat[i, col], std_mat[i, col])
        )
        rank = 1
        i = 0
        while i < m:
            j = i
            # 查找并列区间
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


if __name__ == "__main__":
    # 获取问题实例列表（以第一个方法目录为参照）
    ref_dir = os.path.join(TEST_DIR, METABBO_LIST[0])
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(
            f"测试结果目录不存在：{ref_dir}\n"
            "请先运行 for_review/ch3_test.py"
        )

    problem_list = sorted(
        f.split('.')[0]
        for f in os.listdir(ref_dir) if f.endswith('.pkl')
    )
    print(f"检测到 {len(problem_list)} 个问题实例：{problem_list[:3]} ...")

    m = len(BASELINE_LIST)
    n = len(problem_list)
    mean_mat = np.full((m, n), np.nan)
    std_mat  = np.full((m, n), np.nan)

    for i, method in enumerate(BASELINE_LIST):
        for j, problem in enumerate(problem_list):
            pkl = os.path.join(TEST_DIR, method, f"{problem}.pkl")
            if not os.path.exists(pkl):
                print(f"  [WARN] 结果文件缺失: {method}/{problem}.pkl")
                continue
            mean_mat[i, j], std_mat[i, j] = load_final_cost(pkl)
        print(f"  已处理: {method}")

    rank_mat = compute_ranks(mean_mat, std_mat)

    # 平均排名 & 最终排名
    avg_rank = np.nanmean(rank_mat, axis=1)
    sorted_idx = np.argsort(avg_rank)
    final_rank = np.empty(m, dtype=int)
    final_rank[sorted_idx] = np.arange(1, m + 1)

    # ── 导出 Excel ──
    df_mean = pd.DataFrame(mean_mat, index=BASELINE_LIST, columns=problem_list)
    df_std  = pd.DataFrame(std_mat,  index=BASELINE_LIST, columns=problem_list)
    df_rank = pd.DataFrame(rank_mat, index=BASELINE_LIST, columns=problem_list)
    df_rank['avg_rank']   = avg_rank
    df_rank['final_rank'] = final_rank

    # Table 3-1：mean ± std 合并展示
    combined = {}
    for prob in problem_list:
        col_mean = df_mean[prob]
        col_std  = df_std[prob]
        combined[prob] = [
            f"{m:.4e} ± {s:.2e}" for m, s in zip(col_mean, col_std)
        ]
    df_combined = pd.DataFrame(combined, index=BASELINE_LIST)
    df_combined['avg_rank']   = avg_rank.round(2)
    df_combined['final_rank'] = final_rank

    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df_combined.to_excel(writer, sheet_name='Table3-1 (mean±std+rank)')
        df_mean.to_excel(writer,    sheet_name='mean (raw)')
        df_std.to_excel(writer,     sheet_name='std (raw)')
        df_rank.to_excel(writer,    sheet_name='Table3-2 (rank)')

    print(f"\nTable 3-1 / 3-2 已保存至 {OUTPUT_FILE}")
