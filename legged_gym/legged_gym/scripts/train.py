import numpy as np
import os
from datetime import datetime
import sys
# Ensure imports work from any cwd by adding the repo's `legged_gym` root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import isaacgym  
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch
# import debugpy
# print('waiting for debugpy ...')
# debugpy.listen(1990)
# debugpy.wait_for_client()
# print("Debugpy attached, please enter 'F5' for debuging...")


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
