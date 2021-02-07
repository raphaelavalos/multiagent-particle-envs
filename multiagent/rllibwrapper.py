# From https://github.com/wsjeon/maddpg-rllib/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.spaces import Discrete, Box, MultiDiscrete, Dict
from ray import rllib
from make_env import make_env

import numpy as np
import time


class RLlibMultiAgentParticleEnv(rllib.MultiAgentEnv):
    """Wraps OpenAI Multi-Agent Particle env to be compatible with RLLib multi-agent."""

    def __init__(self, **mpe_args):
        """Create a new Multi-Agent Particle env compatible with RLlib.
        Arguments:
            mpe_args (dict): Arguments to pass to the underlying
                make_env.make_env instance.
        Examples:
            >>> from multiagent.rllibwrapper import RLlibMultiAgentParticleEnv
            >>> env = RLlibMultiAgentParticleEnv(scenario_name="simple_reference")
            >>> print(env.reset())
        """

        self._env = make_env(**mpe_args)
        self.num_agents = self._env.n
        self.agent_ids = list(range(self.num_agents))

        self.observation_space_dict = self._make_dict(self._env.observation_space)
        self.action_space_dict = self._make_dict(self._env.action_space)

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs_dict: New observations for each ready agent.
        """

        obs_dict = self._make_dict(self._env.reset())
        return obs_dict

    def step(self, action_dict):
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns:
            obs_dict:
                New observations for each ready agent.
            rew_dict:
                Reward values for each ready agent.
            done_dict:
                Done values for each ready agent.
                The special key "__all__" (required) is used to indicate env termination.
            info_dict:
                Optional info values for each agent id.
        """

        actions = [action_dict[agent_id] for agent_id in self.agent_ids]
        obs_list, rew_list, done_list, info_list = self._env.step(actions)

        obs_dict = self._make_dict(obs_list)
        rew_dict = self._make_dict(rew_list)
        done_dict = self._make_dict(done_list)
        done_dict["__all__"] = all(done_list)
        # FIXME: Currently, this is the best option to transfer agent-wise termination signal without touching RLlib code hugely.
        # FIXME: Hopefully, this will be solved in the future.
        info_dict = self._make_dict([{"done": done} for done in done_list])

        return obs_dict, rew_dict, done_dict, info_dict

    def seed(self, seed):
        self._env.seed(seed)

    def render(self, mode='human'):
        time.sleep(0.05)
        return self._env.render(mode=mode)

    def _make_dict(self, values):
        return dict(zip(self.agent_ids, values))


class RLLibSingleAgentParticleEnv(gym.Env):
    def __init__(self, **mpe_args):
        self._env = make_env(**mpe_args)
        self.num_agents = self._env.n
        obs_shape = (self.num_agents,) + self._env.observation_space[0].shape
        self.nbr_actions = np.prod(self._env.action_space[0].shape)
        obs_dict = {"obs_agent_{}".format(i): obs for i, obs in enumerate(self._env.observation_space)}
        self.observation_space = Dict({
            **obs_dict,
            "prev_reward": Box(low=-np.inf, high=np.inf, shape=(self.num_agents,))
        })
        self.action_space = Box(low=self._env.action_space[0].low.min(),
                                high=self._env.action_space[0].high.max(),
                                shape=(self.num_agents * self.nbr_actions,))

    def _obs(self, obs, prev_reward=None):
        obs_ = {"obs_agent_{}".format(i): obs for i, obs in enumerate(obs)}
        if prev_reward is None:
            prev_reward = np.zeros((self.num_agents,))
        else:
            prev_reward = np.array(prev_reward)
        obs_["prev_reward"] = prev_reward
        return obs_

    def reset(self):
        return self._obs(self._env.reset())

    def step(self, action):
        action = action.reshape((self.num_agents, self.nbr_actions))
        action = [action[i] for i in range(self.num_agents)]
        obs_list, rew_list, done_list, info_list = self._env.step(action)
        obs = self._obs(obs_list, rew_list)
        info = {"reward": rew_list, **info_list}
        done = np.all(done_list)
        reward = np.mean(rew_list)
        return obs, reward, done, info

    def seed(self, seed=None):
        return self._env.seed()

