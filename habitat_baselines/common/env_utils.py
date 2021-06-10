#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import subprocess
import random
from typing import Type, Union
from .param import args

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, ThreadedVectorEnv, make_dataset

from habitat.core.logging import logger

def get_sync_dir(file):
    # 只改source_data就好
    source_data = file

    sync_source_dir = os.path.join(args.atp_path, source_data.strip('/'))
    sync_dest_dir = os.path.join(args.host_path,
                                 os.path.dirname(source_data.strip('/')))

    logger.info(f"syncing {sync_source_dir} {sync_dest_dir}")
    # 确保同步目录存在, 防止拷贝文件时异常
    if not os.path.exists(sync_dest_dir):
        cmd_line = "mkdir -p {0}".format(sync_dest_dir)
        subprocess.call(cmd_line.split())

    data_dir = os.path.join(args.host_path, source_data.strip('/'))

    if not os.path.exists(data_dir):
        # --info=progress2需要rsync3.1+的版本支持
        cmd_line = "rsync -a {0} {1}".format(sync_source_dir, sync_dest_dir)
        subprocess.call(cmd_line.split())

    return data_dir

def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        rank: rank of env to be created (for seeding).

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET, 
        env_config=config.TASK_CONFIG.ENVIRONMENT
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.AGENT_SEED)
    return env

def construct_envs(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created.

    Returns:
        VectorEnv object created according to specification.
    """
    if config.TASK_CONFIG.ENVIRONMENT.MULTI_AGENT:
        batch_size = config.BATCH_SIZE
        agent_num = config.TASK_CONFIG.ENVIRONMENT.AGENT_NUM
        num_processes = batch_size * agent_num
    else:
        batch_size = config.NUM_PROCESSES
        num_processes = batch_size
        agent_num = 1
    configs = []
    _config = config.TASK_CONFIG.DATASET
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(_config.TYPE)
    scenes = _config.CONTENT_SCENES
    if args.atp:
        _config["SCENES_DIR"] = get_sync_dir(_config["SCENES_DIR"])
        _config["CUR_DATA_PATH"] = _config["DATA_PATH"].format(split=_config.SPLIT)
        get_sync_dir(os.path.join(os.path.dirname(_config["CUR_DATA_PATH"]), "content/")) # sync content
        _config["CUR_DATA_PATH"] = get_sync_dir(_config["CUR_DATA_PATH"]) # sync data
    else:
        _config["CUR_DATA_PATH"] = _config["DATA_PATH"].format(split=_config.SPLIT)
    if "*" in _config.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(_config)
    logger.info(scenes)
    if batch_size > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        # if len(scenes) < batch_size:   # Allowing more workers per scene
        #     raise RuntimeError(
        #         "reduce the number of processes as there "
        #         "aren't enough number of scenes"
        #     )

        random.shuffle(scenes)

    scene_splits = [[] for _ in range(batch_size)]

    if len(scenes) >= batch_size:
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)
    else:
        sc=0
        for i in range(batch_size):
            scene_splits[i].append(scenes[sc%len(scenes)])
            sc+=1



    # assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.GROUP_SEED = task_config.SEED + i // agent_num
        task_config.AGENT_SEED = task_config.SEED + i
        task_config.pop("SEED", None)

        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i//agent_num]

        gpu_num = len(config.SIMULATOR_GPU_ID)
        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID[i%gpu_num]
        )
                             
        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        
        if config.TASK_CONFIG.ENVIRONMENT.MULTI_AGENT:
            proc_config.TASK_CONFIG.ENVIRONMENT.GROUP_ID = i // agent_num
            proc_config.TASK_CONFIG.ENVIRONMENT.AGENT_ID = i % agent_num
        
        proc_config.freeze()
        configs.append(proc_config)
    if config.TASK_CONFIG.ENVIRONMENT.MULTI_AGENT:
        envs = habitat.MultiAgentEnv(
            make_env_fn=make_env_fn,
            env_fn_args=tuple(tuple(zip(configs, env_classes))),
            batch_size=batch_size,
            agent_num=agent_num,
            config=config
        )
    else:
        envs = habitat.VectorEnv(
            make_env_fn=make_env_fn,
            env_fn_args=tuple(tuple(zip(configs, env_classes))),
        )
    return envs
