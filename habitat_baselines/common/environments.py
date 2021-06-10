#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.logging import logger


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self._subsuccess_measure_name = self._rl_config.SUBSUCCESS_MEASURE
        if hasattr(self._core_env_config.ENVIRONMENT, "GROUP_ID"):
            self.group_id = self._core_env_config.ENVIRONMENT.GROUP_ID
        if hasattr(self._core_env_config.ENVIRONMENT, "AGENT_ID"):
            self.agent_id = self._core_env_config.ENVIRONMENT.AGENT_ID


        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        raise NotImplementedError
        # return (
        #     self._rl_config.SLACK_REWARD - 1.0,
        #     self._rl_config.SUCCESS_REWARD + 1.0,
        # )

    def get_reward(self, observations, **kwargs):
        raise NotImplementedError
        # current_measure = self._env.get_metrics()[self._reward_measure_name]
        # # logger.info(f"SLACK_REWARD {self.group_id} {self.agent_id} {self._rl_config.SLACK_REWARD}")
        # # logger.info(f"_previous_measure {self.group_id} {self.agent_id} {self._previous_measure}")
        # # logger.info(f"current_measure {self.group_id} {self.agent_id} {current_measure}")
        # # logger.info(f"SUCCESS_REWARD {self.group_id} {self.agent_id} {self._episode_success()} {self._rl_config.SUCCESS_REWARD}")
        # # logger.info(f"SUBSUCCESS_REWARD {self.group_id} {self.agent_id} {self._episode_subsuccess()} {self._rl_config.SUBSUCCESS_REWARD}")
        # reward = self._rl_config.SLACK_REWARD

        # if self._episode_subsuccess():
        #     current_measure = self._env.task.foundDistance

        # reward += self._previous_measure - current_measure
        # self._previous_measure = current_measure

        # if self._episode_subsuccess():
        #     self._previous_measure = self._env.get_metrics()[self._reward_measure_name]

        # if self._episode_success():
        #     reward += self._rl_config.SUCCESS_REWARD
        # elif self._episode_subsuccess():
        #     reward += self._rl_config.SUBSUCCESS_REWARD
        # elif self._env.task.is_found_called and self._rl_config.FALSE_FOUND_PENALTY:
        #     reward -= self._rl_config.FALSE_FOUND_PENALTY_VALUE
        # # logger.info(f"reward {self.group_id} {self.agent_id} {self._env.task.is_found_called} {reward}")
        # return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]


    def _episode_subsuccess(self):
        return self._env.get_metrics()[self._subsuccess_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        info =  self.habitat_env.get_metrics()
        info["is_found_called"] = self.habitat_env._task.is_found_called
        if self._episode_subsuccess():
            info["foundDistance"] = self.habitat_env._task.foundDistance
        else:
            info["foundDistance"] = None
        info["currGoalIndex"] = self.habitat_env._task.currGoalIndex
        return info
