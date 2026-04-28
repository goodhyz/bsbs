"""
第三章测试脚本
测试所有参评方法在 7 个问题集上的性能，结果保存至 test/{problem}/metadata/
同时测试 checkpoint-0 版本（初始模型），用于计算学习效率指标。

运行目录：MetaBox 项目根目录
    python for_review/ch3_test.py
"""

import os
from metaevobox import Tester, Config, construct_problem_set, get_baseline
from metaevobox.baseline.metabbo import *
from metaevobox.baseline.bbo import *
from metaevobox.environment.optimizer import *

# ── 方法列表（第三章，已去除 GLHF/B2OPT）──
METABBO_LIST = [
    'RNNOPT', 'DEDDQN', 'DEDQN', 'LDE',
    'RLPSO', 'RLEPSO', 'NRLPSO', 'LES',
    'GLEET', 'RLDAS', 'SYMBOL', 'RLDEAFL',
]
BBO_LIST = ['PSO', 'DE', 'SHADE', 'JDE21', 'MADDE']

# 不支持分布外（非 10D）测试的方法
NO_ZERO_SHOT = {'RNNOPT', 'RLPSO', 'RLDAS'}

# 预训练模型目录（checkpoint-20 等价物）
MODEL_DIR = "src/metaevobox/model/bbob-10D/difficult"

# 测试问题集配置
TEST_PROBLEMS = [
    ('bbob-10D',       'difficult', 16),
    ('bbob-noisy-10D', 'all',       16),
    ('bbob-30D',       'all',       16),
    ('bbob-noisy-30D', 'all',       16),
    ('protein',        'all',       64),
    ('uav',            'all',       56),
    ('hpo-b',          'all',       64),
]


def run_test(problem, difficulty, batch_size, log_dir, checkpoint_suffix=""):
    """在单个问题集上运行所有方法的测试."""
    is_10D = "10D" in problem

    config = {
        'test_problem':    problem,
        'test_difficulty': difficulty,
        'test_batch_size': batch_size,
        'full_meta_data':  True,
        'log_dir':         log_dir,
        'baselines':       {},
    }

    for baseline in METABBO_LIST:
        if not is_10D and baseline in NO_ZERO_SHOT:
            continue  # 跳过不支持零样本迁移的方法
        model_path = os.path.join(MODEL_DIR, f"{baseline}{checkpoint_suffix}.pkl")
        if not os.path.exists(model_path):
            print(f"  [WARN] 模型文件不存在，跳过 {baseline}: {model_path}")
            continue
        config['baselines'][baseline] = {
            'agent':           baseline,
            'optimizer':       eval(f"{baseline}_Optimizer"),
            'model_load_path': model_path,
        }

    for baseline in BBO_LIST:
        config['baselines'][baseline] = {
            'optimizer': eval(f"{baseline}"),
        }

    cfg = Config(config)
    cfg.test_log_dir = cfg.log_dir + problem
    cfg, datasets = construct_problem_set(cfg)
    baselines, cfg = get_baseline(cfg)
    tester = Tester(cfg, baselines, datasets)
    tester.test(log=False)
    print(f"  完成: {log_dir}{problem}")


if __name__ == "__main__":
    # ── 阶段 1：测试 checkpoint-20（最终模型）──
    print("=" * 60)
    print("阶段 1：测试最终模型（checkpoint-20）")
    print("=" * 60)
    for problem, difficulty, batch_size in TEST_PROBLEMS:
        print(f"测试 {problem} ...")
        run_test(problem, difficulty, batch_size, log_dir="test/")

    print()

    # ── 阶段 2：测试 checkpoint-0（初始模型），用于计算学习效率 ──
    # 需要 src/metaevobox/model/bbob-10D/difficult/{baseline}-checkpoint-0.pkl 存在
    # 如果不存在则跳过此阶段
    print("=" * 60)
    print("阶段 2：测试初始模型（checkpoint-0，用于学习效率计算）")
    print("=" * 60)
    cp0_exists = any(
        os.path.exists(os.path.join(MODEL_DIR, f"{b}-checkpoint-0.pkl"))
        for b in METABBO_LIST
    )
    if not cp0_exists:
        print("[INFO] 未找到 checkpoint-0 模型文件，跳过学习效率测试。")
        print("       若需计算学习效率，请将各方法 checkpoint-0.pkl 放至：")
        print(f"       {MODEL_DIR}/{{method}}-checkpoint-0.pkl")
    else:
        print("测试 bbob-10D (checkpoint-0) ...")
        run_test('bbob-10D', 'difficult', 16,
                 log_dir="test/checkpoint-0/", checkpoint_suffix="-checkpoint-0")

    print()
    print("第三章测试完毕。结果目录：test/")
