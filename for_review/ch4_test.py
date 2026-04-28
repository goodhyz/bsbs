"""
第四章测试脚本
测试 RLDEAFL_T（Trans-RLDE-AFL）及第四章选定对比方法在 7 个问题集上的性能。
结果保存至 test/{problem}/metadata/（与第三章共享目录，方法名为 key）。

运行目录：MetaBox 项目根目录
    python for_review/ch4_test.py
"""

import os
from metaevobox import Tester, Config, construct_problem_set, get_baseline
from metaevobox.baseline.metabbo import *
from metaevobox.baseline.bbo import *
from metaevobox.environment.optimizer import *

# ── 第四章测试方法 ──
# RLDEAFL、GLEET、RLDAS、LDE 的结果已由 ch3_test.py 生成，此处仅补充新方法
NEW_METHODS = ['RLDEAFL_T']

# 第四章全部方法（含已有 ch3 结果）
CH4_METABBO_LIST = ['RLDEAFL_T', 'RLDEAFL', 'GLEET', 'RLDAS', 'LDE']
CH4_BBO_LIST     = ['DE', 'SHADE', 'JDE21', 'MADDE']

NO_ZERO_SHOT = {'RLDAS'}  # 第四章方法中仅 RLDAS 不支持零样本

MODEL_DIR = "src/metaevobox/model/bbob-10D/difficult"

TEST_PROBLEMS = [
    ('bbob-10D',       'difficult', 16),
    ('bbob-noisy-10D', 'all',       16),
    ('bbob-30D',       'all',       16),
    ('bbob-noisy-30D', 'all',       16),
    ('protein',        'all',       64),
    ('uav',            'all',       56),
    ('hpo-b',          'all',       64),
]


def run_test_new_methods(problem, difficulty, batch_size):
    """仅测试新增方法（已有结果的方法不重复测试）."""
    is_10D = "10D" in problem

    config = {
        'test_problem':    problem,
        'test_difficulty': difficulty,
        'test_batch_size': batch_size,
        'full_meta_data':  True,
        'log_dir':         'test/',
        'baselines':       {},
    }

    for baseline in NEW_METHODS:
        if not is_10D and baseline in NO_ZERO_SHOT:
            continue
        model_path = os.path.join(MODEL_DIR, f"{baseline}.pkl")
        if not os.path.exists(model_path):
            print(f"  [ERROR] 模型文件不存在: {model_path}")
            print("          请先运行 for_review/train_rldeafl_t.py 并将最终检查点")
            print(f"          复制至 {model_path}")
            continue
        config['baselines'][baseline] = {
            'agent':           baseline,
            'optimizer':       eval(f"{baseline}_Optimizer"),
            'model_load_path': model_path,
        }

    if not config['baselines']:
        print(f"  [SKIP] {problem}：无可用模型")
        return

    cfg = Config(config)
    cfg.test_log_dir = cfg.log_dir + problem
    cfg, datasets = construct_problem_set(cfg)
    baselines, cfg = get_baseline(cfg)
    tester = Tester(cfg, baselines, datasets)
    tester.test(log=False)
    print(f"  完成: test/{problem}")


if __name__ == "__main__":
    print("=" * 60)
    print("第四章：测试 RLDEAFL_T（Trans-RLDE-AFL）")
    print("=" * 60)
    print("注意：RLDEAFL/GLEET/RLDAS/LDE 的结果直接复用 ch3_test.py 输出，")
    print("      此脚本仅补充 RLDEAFL_T 的测试结果。")
    print()

    for problem, difficulty, batch_size in TEST_PROBLEMS:
        print(f"测试 {problem} ...")
        run_test_new_methods(problem, difficulty, batch_size)

    print()
    print("第四章测试完毕。结果与第三章共享目录：test/")
