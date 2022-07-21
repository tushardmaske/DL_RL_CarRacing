import collections
from collections import namedtuple
import numpy as np
import os
import gzip
import pickle

import torch


class ReplayBuffer:

    # implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, capacity=10000):
        self._data = collections.deque(maxlen=capacity)
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)

        batch_states = torch.from_numpy(np.array([self._data.states[i] for i in batch_indices])).cuda()
        batch_actions = torch.from_numpy(np.array([self._data.actions[i] for i in batch_indices])).cuda()
        batch_next_states = torch.from_numpy(np.array([self._data.next_states[i] for i in batch_indices])).cuda()
        batch_rewards = torch.from_numpy(np.array([self._data.rewards[i] for i in batch_indices])).cuda()
        batch_dones = torch.from_numpy(np.array([self._data.dones[i] for i in batch_indices])).cuda()
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

    def __len__(self):
        """
        This method returns the length of the replay buffer
        """
        return len(self._data.states)
