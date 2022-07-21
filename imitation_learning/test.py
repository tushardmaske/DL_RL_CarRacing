from __future__ import print_function

import sys

sys.path.append("../")

from datetime import datetime
import numpy as np
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_episode(env, agent, hist, rendering=True, max_timesteps=1000):

    # Save history
    image_hist = []

    episode_reward = 0
    step = 0

    state = env.reset()

    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events()

    state = rgb2gray(state)

    image_hist.extend([state] * (hist + 1))
    state = np.array(image_hist).reshape(1, hist + 1, 96, 96)
    i = 0
    action_arr = 0
    while True:

        # preprocess the state in the same way than in your preprocessing in train_agent.py

        # get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration

        a = np.argmax(agent.predict(state).cpu().detach().numpy())
        action_arr = id_to_action(a)

        # Clip the acceleration in the start
        # if hist > 0:
        if i < 1000 and a == ACCELERATE:
            action_arr = softmax(id_to_action(a)) * [0.0, 1.0, 0.0]
            i += 1

        # print("Action array : ", action_arr)
        next_state, r, done, info = env.step(action_arr)

        next_state = rgb2gray(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(1, hist + 1, 96, 96)

        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15  # number of episodes to test
    hs = 4
    # load agent
    agent = BCAgent(history_length=hs)
    # agent.load("models/bc_agent.pt")
    agent.load("models/agent_HL_5.pt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, hist=hs, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')