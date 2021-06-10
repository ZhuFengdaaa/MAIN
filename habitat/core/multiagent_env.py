#!/usr/bin/env python3
                                     
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
from gym import Space, spaces
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import gym
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict

import habitat
from habitat.config import Config
from habitat.core.env import Env, Observations, RLEnv
from habitat.core.logging import logger
from habitat.core.utils import tile_images
from habitat.core.spaces import ActionSpace, EmptySpace

try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
CALL_COMMAND = "call"
EPISODE_COMMAND = "current_episode"
EPISODES_COMMAND = "current_episodes"
SCENE_SWITCH_COMMAND = "scene_switch"



def _make_env_fn(
    config: Config, dataset: Optional[habitat.Dataset] = None, rank: int = 0
) -> Env:
    """Constructor for default habitat `env.Env`.

    :param config: configuration for environment.
    :param dataset: dataset for environment.
    :param rank: rank for setting seed of environment
    :return: `env.Env` / `env.RLEnv` object
    """
    habitat_env = Env(config=config, dataset=dataset)
    habitat_env.seed(config.SEED + rank)
    return habitat_env

class MAMONTask:
    def __init__(self, batch_size, agent_num, rl_config):
        self.distance_to_currgoal = None
        self.batch_size = batch_size
        self.agent_num = agent_num
        self._rl_config = rl_config
        self._reward_measure_name = rl_config.REWARD_MEASURE
        self._success_measure_name = rl_config.SUCCESS_MEASURE
        self._subsuccess_measure_name = rl_config.SUBSUCCESS_MEASURE
    
    def update_metric(self, info, index_env):
        if len(index_env) == self.batch_size * self.agent_num:
            self.prev_measures = np.array([_info[self._reward_measure_name] for _info in info])
        else:
            prev_measures = np.array([_info[self._reward_measure_name] for _info in info])
            for i, idx in enumerate(index_env):
                self.prev_measures[idx] = prev_measures[i]
    
    def global_reward(self, infos):
        rewards = np.full((self.batch_size), self._rl_config.SLACK_REWARD)
        current_measures = [_info[self._reward_measure_name] for _info in infos]
        for i, info in enumerate(infos):
            if info["sub_success"]:
                current_measures[i] = info["foundDistance"]

        global_current_measures = self.global_min_metric(current_measures)
        global_prev_measures = self.global_min_metric(self.prev_measures)

        rewards += global_prev_measures - global_current_measures

        # success reward
        global_success_measures = self.global_success_metric(infos)
        rewards += global_success_measures
        return rewards

    def global_success_metric(self, infos):
        global_measures = np.zeros(self.batch_size)
        success_measures = self.global_max_metric([_info["success"] for _info in infos])
        subsuccess_measures = self.global_max_metric([_info["sub_success"] for _info in infos])
        call_found_measures = self.global_max_metric([_info["is_found_called"] for _info in infos])
        global_measures += self._rl_config.SUCCESS_REWARD * np.array(success_measures)
        global_measures += self._rl_config.SUBSUCCESS_REWARD * \
            (np.logical_or(np.array(success_measures), np.array(subsuccess_measures)).astype(int) - np.array(success_measures))
        # global_measures -= self._rl_config.FALSE_FOUND_PENALTY_VALUE * np.array(call_found_measures)
        return global_measures
       

    def global_max_metric(self, measures):
        measures = np.array(measures)
        assert len(measures) == self.batch_size*self.agent_num
        global_measures = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            global_measures[i] = np.max(measures[self.agent_num*i: self.agent_num*(i+1)])
        return global_measures
    
    def global_min_metric(self, measures):
        measures = np.array(measures)
        assert len(measures) == self.batch_size*self.agent_num
        global_measures = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            global_measures[i] = np.min(measures[self.agent_num*i: self.agent_num*(i+1)])
        return global_measures


class MultiAgentEnv:
    r"""Vectorized environment which creates multiple processes where each
    process runs its own environment. Main class for parallelization of
    training and evaluation.


    All the environments are synchronized on step and reset methods.
    """

    observation_spaces: List[SpaceDict]
    action_spaces: List[SpaceDict]
    _workers: List[Union[mp.Process, Thread]]
    _is_waiting: bool
    _num_envs: int
    _auto_reset_done: bool
    _mp_ctx: BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(
        self,
        batch_size, agent_num, 
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
        env_fn_args: Sequence[Tuple] = None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
        config=None
    ) -> None:
        """..

        :param make_env_fn: function which creates a single environment. An
            environment can be of type `env.Env` or `env.RLEnv`
        :param env_fn_args: tuple of tuple of args to pass to the
            `_make_env_fn`.
        :param auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        :param multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            :py:`{'spawn', 'forkserver', 'fork'}`; :py:`'forkserver'` is the
            recommended method as it works well with CUDA. If :py:`'fork'` is
            used, the subproccess must be started before any other GPU useage.
        """
        self._is_waiting = False
        self._is_closed = True

        assert (
            env_fn_args is not None and len(env_fn_args) > 0
        ), "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)
        self.batch_size = batch_size
        self.agent_num = agent_num

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        logger.info(f"multiprocessing_start_method {multiprocessing_start_method}")
        logger.info(f"self._valid_start_methods {self._valid_start_methods}")
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
            self._connection_poll_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args, make_env_fn
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))
        _observation_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        self.rollout_observation_spaces = _observation_spaces[0]
        self.observation_spaces = self.process_multi_agent_space(_observation_spaces)
        
        for write_fn in self._connection_write_fns:
            write_fn((ACTION_SPACE_COMMAND, None))
        _action_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        self.rollout_action_spaces = _action_spaces[0]
        self.action_spaces = self.process_multi_agent_space(_action_spaces)

        self._obs = np.array([None] * self._num_envs)
        self._dones = np.array([False] * self._num_envs)
    
        self.config = config
        self._task = MAMONTask(batch_size, agent_num, rl_config=config.RL)

        # self._paused = []

    def process_multi_agent_space(self, _space):
        assert len(_space) % self.agent_num == 0
        _space = _space[0]
        if _space.__class__.__name__ == "Dict":
            return_space = {}
            for k, v in _space.spaces.items():
                if type(v) == spaces.Box:
                    _low = v.low
                    _high = v.high
                    assert np.all(_low == _low[0])
                    assert np.all(_high == _high[0])
                    _low = _low.flat[0].item()
                    _high = _high.flat[0].item()
                    _shape = v.shape
                    _dtype = v.dtype
                    _shape = list(_shape)
                    _shape[-1] = _shape[-1] * self.agent_num
                    _shape = tuple(_shape)
                    return_space[k] = type(v)(_low, _high, _shape, _dtype)
            return_space = SpaceDict(return_space)
        elif _space.__class__.__name__ == "ActionSpace":
            return_space = _space
        else:
            raise NotImplementedError
        return return_space


    @property
    def num_envs(self):
        r"""number of individual environments.
        """
        # return self._num_envs - len(self._paused)
        return self._num_envs

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
        group_id = None,
        agent_id = None
    ) -> None:
        r"""process worker for creating and interacting with the environment.
        """
        env = env_fn(*env_fn_args)
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # different step methods for habitat.RLEnv and habitat.Env
                    if isinstance(env, habitat.RLEnv) or isinstance(
                        env, gym.Env
                    ):
                        # habitat.RLEnv
                        observations, reward, done, info = env.step(**data)
                        ### cannot self-reset in multi-agent
                        # if auto_reset_done and done:
                        #     observations = env.reset()
                        connection_write_fn([observations, reward, done, info])
                    elif isinstance(env, habitat.Env):
                        # habitat.Env
                        observations = env.step(**data)
                        ### cannot self-reset in multi-agent
                        # if auto_reset_done and env.episode_over:
                        #     observations = env.reset()
                        connection_write_fn(observations)
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    observations, do_switch, info = env.reset()
                    connection_write_fn((observations, do_switch, info))

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif (
                    command == OBSERVATION_SPACE_COMMAND
                    or command == ACTION_SPACE_COMMAND
                ):
                    if isinstance(command, str):
                        connection_write_fn(getattr(env, command))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(env, function_name)()
                    else:
                        result = getattr(env, function_name)(**function_args)
                    connection_write_fn(result)

                # TODO: update CALL_COMMAND for getting attribute like this
                elif command == EPISODE_COMMAND:
                    connection_write_fn(env.current_episode)
                elif command == EPISODES_COMMAND:
                    connection_write_fn(env.episodes)
                elif command == SCENE_SWITCH_COMMAND:
                    result = env.forced_scene_switch()
                    connection_write_fn(result)
                else:
                    raise NotImplementedError

                command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            env.close()

    def _spawn_workers(
        self,
        env_fn_args: Sequence[Tuple],
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)]
        )
        self._workers = []
        for idx, (worker_conn, parent_conn, env_args) in enumerate(zip(
            worker_connections, parent_connections, env_fn_args
        )):
            group_id = idx // self.agent_num
            agent_id = idx % self.agent_num
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(               
                    worker_conn.recv,
                    worker_conn.send,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                    worker_conn,
                    parent_conn,
                    group_id,
                    agent_id
                ),
            )
            self._workers.append(ps)
            ps.daemon = True
            ps.start()
            worker_conn.close()
        return (
            [p.recv for p in parent_connections],
            [p.send for p in parent_connections],
            [p.poll for p in parent_connections],
        )

    def current_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((EPISODE_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results
    
    def get_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((EPISODES_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def check_group_correct(self, name, infos=None):
        if name == "reset":
            env_episodes = self.get_episodes()
            assert len(env_episodes) % self.agent_num == 0, f"agent num supposed to be {self.agent_num} not {len(env_episodes)}"
            meta_episode_ids = None
            for i, episode in enumerate(env_episodes):
                episode_ids = np.array([_epsode.episode_id for _epsode in episode])
                if i % self.agent_num == 0:
                    meta_episode_ids = episode_ids
                else:
                    assert np.array_equal(meta_episode_ids, episode_ids)

        results = self.current_episodes()
        assert len(results) % self.agent_num == 0, f"agent num supposed to be {self.agent_num} not {len(results)}"
        for i, result in enumerate(results):
            if i % self.agent_num == 0:
                episode_id = result.episode_id
                goals = result.goals
            else:
                assert episode_id == result.episode_id
                assert goals == result.goals
    
        if infos is not None:
            # TODO: align task currGoalIndex inside a group is required
            for bs in range(self.batch_size):
                _goal_idx = infos[bs*self.agent_num]["currGoalIndex"]
                for a_i in range(self.agent_num):
                    assert _goal_idx == infos[bs*self.agent_num]["currGoalIndex"]
        return True

    def reset(self):
        return self.reset_at(index_env=list(range(self._num_envs)))

    def _forced_scene_switch_if(self, do_switchs, index_env):
        _index_env = []
        last_group = -1
        assert len(do_switchs) == len(index_env)
        for i, env_id in enumerate(index_env):
            if last_group == env_id // self.agent_num:
                continue
            if do_switchs[i] == True:
                __index_env = list(range(env_id // self.agent_num * self.agent_num\
                    , (env_id // self.agent_num + 1) * self.agent_num))
                _index_env += __index_env
                last_group = env_id // self.agent_num
        
        for _idx in _index_env:
            self._connection_write_fns[_idx]((SCENE_SWITCH_COMMAND, None))
        _results = []
        for _idx in _index_env:
            _results.append(self._connection_read_fns[_idx]())
        return _results, _index_env

    
    def reset_at(self, index_env: list, set_waiting=True):
        r"""Reset in the index_env environment in the vector.

        :param index_env: index of the environment to be reset
        :return: list containing the output of reset method of indexed env.
        """
        if len(index_env) == 0:
            return []
        if set_waiting:
            self._is_waiting = True
        assert type(index_env) == list
        for _idx in index_env:
            self._connection_write_fns[_idx]((RESET_COMMAND, None))
        results = []
        for _idx in index_env:
            results.append(self._connection_read_fns[_idx]())
        unzip_results = list(zip(*results))
        observations = list(unzip_results[0])
        do_switchs = list(unzip_results[1])
        infos = list(unzip_results[2])
        _results, _index_env = self._forced_scene_switch_if(do_switchs, index_env) # observations, do_switchs, infos
        if len(_index_env) > 0:
            _unzip_results = list(zip(*_results))
            _observations = list(_unzip_results[0])
            _infos = list(_unzip_results[2])
            for i, _idx in enumerate(_index_env):
                idx = index_env.index(_idx)
                observations[idx] = _observations[i]
                infos[idx] = _infos[i]
        if set_waiting:
            self._is_waiting = False
        self._task.update_metric(infos, index_env)
        return observations

    def step_at(self, index_env: int, action: Dict[str, Any]):
        r"""Step in the index_env environment in the vector.

        :param index_env: index of the environment to be stepped into
        :param action: action to be taken
        :return: list containing the output of step method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((STEP_COMMAND, action))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def async_step(self, data: List[Union[int, str, Dict[str, Any]]]) -> None:
        r"""Asynchronously step in the environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        """
        # Backward compatibility
        if isinstance(data[0], (int, np.integer, str)):
            data = [{"action": {"action": action}} for action in data]

        self._is_waiting = True
        for i, (write_fn, args) in enumerate(zip(self._connection_write_fns, data)):
            write_fn((STEP_COMMAND, args))
    
    def wait_step(self) -> List[Observations]:
        r"""Wait until all the asynchronized environments have synchronized.
        """
        observations = []
        for i, read_fn in enumerate(self._connection_read_fns):
            assert self._dones[i] == False
            _obs = read_fn()
            self._dones[i] = _obs[2] or self._dones[i]
            observations.append(_obs)
            self._obs[i] = _obs

        global_dones = []
        # calculate reward
        infos = [obs[3] for obs in observations]
        rewards = self._task.global_reward(infos)
        # reset
        reset_list = []
        for _bs in range(self.batch_size):
            reset_flag = np.any(self._dones[_bs*self.agent_num: (_bs+1)*self.agent_num])
            global_dones.append(reset_flag)
            if reset_flag:
                reset_list += list(range(_bs*self.agent_num, (_bs+1)*self.agent_num))
                self._dones[_bs*self.agent_num: (_bs+1)*self.agent_num] = False
        results = self.reset_at(reset_list, set_waiting=False)
        for i, reset_id in enumerate(reset_list):
            self._obs[reset_id] = results[i]
            observations[reset_id][0] = self._obs[reset_id] # [0] is obs in [obs, reward, done, info]
        self._is_waiting = False

        update_list = [i for i in range(self.batch_size * self.agent_num) if i not in reset_list]
        update_infos = [infos[idx] for idx in update_list]
        self._task.update_metric(update_infos, update_list)
        check_success = self.check_group_correct("step", infos=infos) # debug only
        return observations, rewards, global_dones, check_success # obs: 16, rewards: 8

    def step(self, data: List[Union[int, str, Dict[str, Any]]]) -> List[Any]:
        r"""Perform actions in the vectorized environments.

        :param data: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        self.async_step(data)
        return self.wait_step()

    def close(self) -> None:
        if self._is_closed:
            return

        index_list = [False for _ in range(len(self._connection_read_fns))]
        
        for idx, (poll_fn, read_fn) in enumerate(zip(self._connection_poll_fns, self._connection_read_fns)):
            try:
                if poll_fn(1):
                    read_fn()
                index_list[idx] = True
            except:
                pass

        for idx, write_fn in enumerate(self._connection_write_fns):
            try:
                write_fn((CLOSE_COMMAND, None))
                index_list[idx] = True
            except:
                pass

        # for _, _, write_fn, _ in self._paused:
        #     write_fn((CLOSE_COMMAND, None))

        for idx, process in enumerate(self._workers):
            if index_list[idx]:
                logger.info(f"process {idx} join")
                try:
                    process.terminate()
                    process.join(10)
                except Exception as e:
                    logger.info(str(e))

        # for idx, (_, _, _, process) in enumerate(self._paused):
        #     logger.info(f"process {idx} join")
        #     process.join()

        self._is_closed = True

    # def pause_at(self, index: int) -> None:
    #     r"""Pauses computation on this env without destroying the env.

    #     :param index: which env to pause. All indexes after this one will be
    #         shifted down by one.

    #     This is useful for not needing to call steps on all environments when
    #     only some are active (for example during the last episodes of running
    #     eval episodes).
    #     """
    #     if self._is_waiting:
    #         for read_fn in self._connection_read_fns:
    #             read_fn()
    #     read_fn = self._connection_read_fns.pop(index)
    #     write_fn = self._connection_write_fns.pop(index)
    #     worker = self._workers.pop(index)
    #     self._paused.append((index, read_fn, write_fn, worker))

    # def resume_all(self) -> None:
    #     r"""Resumes any paused envs.
    #     """
    #     for index, read_fn, write_fn, worker in reversed(self._paused):
    #         self._connection_read_fns.insert(index, read_fn)
    #         self._connection_write_fns.insert(index, write_fn)
    #         self._workers.insert(index, worker)
    #     self._paused = []

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        r"""Calls a function (which is passed by name) on the selected env and
        returns the result.

        :param index: which env to call the function on.
        :param function_name: the name of the function to call on the env.
        :param function_args: optional function args.
        :return: result of calling the function.
        """
        self._is_waiting = True
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        self._is_waiting = False
        return result

    def call(
        self,
        function_names: List[str],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        r"""Calls a list of functions (which are passed by name) on the
        corresponding env (by index).

        :param function_names: the name of the functions to call on the envs.
        :param function_args_list: list of function args for each function. If
            provided, :py:`len(function_args_list)` should be as long as
            :py:`len(function_names)`.
        :return: result of calling the function.
        """
        self._is_waiting = True
        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(
            self._connection_write_fns, func_args
        ):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def render(
        self, mode: str = "human", *args, **kwargs
    ) -> Union[np.ndarray, None]:
        r"""Render observations from all environments in a tiled image.
        """
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, {"mode": "rgb", **kwargs})))
        images = [read_fn() for read_fn in self._connection_read_fns]
        tile = tile_images(images)
        if mode == "human":
            from habitat.core.utils import try_cv2_import

            cv2 = try_cv2_import()

            cv2.imshow("vecenv", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# class ThreadedVectorEnv(VectorEnv):
#     r"""Provides same functionality as `VectorEnv`, the only difference is it
#     runs in a multi-thread setup inside a single process.

#     `VectorEnv` runs in a multi-proc setup. This makes it much easier to debug
#     when using `VectorEnv` because you can actually put break points in the
#     environment methods. It should not be used for best performance.
#     """

#     def _spawn_workers(
#         self,
#         env_fn_args: Sequence[Tuple],
#         make_env_fn: Callable[..., Env] = _make_env_fn,
#     ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
#         parent_read_queues, parent_write_queues = zip(
#             *[(Queue(), Queue()) for _ in range(self._num_envs)]
#         )
#         self._workers = []
#         for parent_read_queue, parent_write_queue, env_args in zip(
#             parent_read_queues, parent_write_queues, env_fn_args
#         ):
#             thread = Thread(
#                 target=self._worker_env,
#                 args=(
#                     parent_write_queue.get,
#                     parent_read_queue.put,
#                     make_env_fn,
#                     env_args,
#                     self._auto_reset_done,
#                 ),
#             )
#             self._workers.append(thread)
#             thread.daemon = True
#             thread.start()
#         return (
#             [q.get for q in parent_read_queues],
#             [q.put for q in parent_write_queues],
#         )
