#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional, Type, Union

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator, ShortestPathPoint
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)


@attr.s(auto_attribs=True, kw_only=True)
class MAMONEpisode:
    r"""Base class for episode specification that includes initial position and
    rotation of agent, scene id, episode.
    :property episode_id: id of episode in the dataset, usually episode number.
    :property scene_id: id of scene in dataset.
    :property start_position: list of length 3 for cartesian coordinates
        :py:`(x, y, z)`.
    :property start_rotation: list of length 4 for (x, y, z, w) elements
        of unit quaternion (versor) representing 3D agent orientation
        (https://en.wikipedia.org/wiki/Versor). The rotation specifying the
        agent's orientation is relative to the world coordinate axes.
    This information is provided by a `Dataset` instance.
    """

    episode_id: str = attr.ib(default=None, validator=not_none_validator)
    scene_id: str = attr.ib(default=None, validator=not_none_validator)
    start_positions: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_rotations: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_position: List[float] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_rotation: List[float] = attr.ib(
        default=None, validator=not_none_validator
    )
    info: Optional[Dict[str, str]] = None
    _shortest_path_cache: Any = attr.ib(init=False, default=None)
    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[ShortestPathPoint]] = None

    object_category: Optional[List[str]] = None
    object_index: Optional[int]
    currGoalIndex: Optional[int] = 0  

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals
        """
        return [f"{os.path.basename(self.scene_id)}_{i}" for i in self.object_category]

    def __getstate__(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"_shortest_path_cache"}
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["_shortest_path_cache"] = None


@registry.register_task(name="MAMONav-v1")
class MultiNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
        Used to explicitly state a type of the task in config.
    """
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None, 
    _config=None) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset, group_id=_config.ENVIRONMENT.GROUP_ID
        , agent_id=_config.ENVIRONMENT.AGENT_ID)
        self.currGoalIndex=0

    def overwrite_sim_config(self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        # return merge_sim_episode_config(sim_config, episode)
        sim_config.defrost()
        sim_config.SCENE = episode.scene_id
        sim_config.freeze()
        if (
            episode.start_position is not None
            and episode.start_rotation is not None
        ):
            agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
            agent_cfg = getattr(sim_config, agent_name)
            agent_cfg.defrost()
            agent_cfg.START_POSITION = episode.start_position
            agent_cfg.START_ROTATION = episode.start_rotation
            agent_cfg.IS_SET_START_STATE = True
            agent_cfg.freeze()
        return sim_config
