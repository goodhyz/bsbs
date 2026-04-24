from metaevobox import Config, Trainer
from metaevobox.baseline.metabbo import RLDEAFL_T
from metaevobox.environment.optimizer import RLDEAFL_T_Optimizer
from metaevobox.environment.problem.utils import construct_problem_set

config = {
    'train_problem': 'bbob-10D',
    'train_difficulty': 'difficult',
    'train_batch_size': 4,
    'train_parallel_mode': 'subproc',
    'device': 'cuda',
    'end_mode': 'epoch',
    'max_epoch': 20,
    'n_checkpoint': 20,
    'agent_save_dir': 'train/',
}

tmp_config = Config(config)
tmp_config.train_name = ""
tmp_config, datasets = construct_problem_set(tmp_config)

metabbo = RLDEAFL_T(tmp_config)
metabbo_opt = RLDEAFL_T_Optimizer(tmp_config)
trainer = Trainer(tmp_config, metabbo, metabbo_opt, datasets)
trainer.train()

print(f"MetaBBO RLDEAFL_T: train/RLDEAFL_T/")
