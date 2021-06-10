#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.insert(0, "")
import random
import numpy as np
import os
import subprocess
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.param import args
from habitat_baselines.config.default import get_config


def main():
    run_exp(**vars(args))

def run_exp(exp_config: str, run_type: str, agent_type: str, opts=None, **kw_args) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    config.defrost()
    config.TRAINER_NAME = agent_type
    config.TASK_CONFIG.TRAINER_NAME = agent_type
    config.freeze()

    if agent_type in ["oracle", "oracle-ego", "no-map"]:
        trainer_init = baseline_registry.get_trainer("oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512 if agent_type=="no-map" else 768
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.5
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
        config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
        if agent_type == "oracle-ego":
            config.TASK_CONFIG.TASK.MEASUREMENTS.append('FOW_MAP')
        config.freeze()
    else:
        trainer_init = baseline_registry.get_trainer("non-oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512
        config.freeze()
        
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()

if __name__ == "__main__":
    main()

    #MIN_DEPTH: 0.5
    #MAX_DEPTH: 5.0
