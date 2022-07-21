# export DISPLAY=:0 

import sys

import utils

sys.path.append("../")

import numpy as np
import gym
from agent.dqn_agent import DQNAgent
from agent.networks import CNN
from tensorboard_evaluation import *
import itertools as it
from utils import *
import matplotlib.pyplot as plt

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)

    # To view the image
    # arr = np.asarray(state)
    # plt.figure()
    # plt.imshow(arr)
    # plt.show()

    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(1, history_length + 1, 96, 96)

    next_state = 0
    r = 0
    terminal = 0
    while True:

        # get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        action_id = agent.act(state=state, deterministic=deterministic)
        action = utils.id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        # To view the image
        # arr = np.asarray(next_state)
        # plt.figure()
        # plt.imshow(arr)
        # plt.show()
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(1, history_length + 1, 96, 96)

        if do_training:
            agent.train(state.squeeze(0), action_id, next_state.squeeze(0), reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, max_timesteps, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), name='CarRacing', stats=["episode_reward", "straight", "left", "right", "accel", "brake"])

    best_reward = 0
    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
       
        stats = run_episode(env, agent, skip_frames=3, max_timesteps=max_timesteps, deterministic=False, do_training=True, rendering=False, history_length=history_length)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.

        eval_episode_reward = []
        if (i + 1) % eval_cycle == 0:
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=False, history_length=history_length)
                eval_episode_reward.append(stats.episode_reward)
            print("Mean of the validation reward ", np.mean(eval_episode_reward))
            max_timesteps = max_timesteps + 50


        # store model.
        if (i + 1) % eval_cycle == 0 or (i + 1) >= num_episodes:
            if best_reward < np.mean(eval_episode_reward):
                # agent.save(os.path.join(model_dir, "dqn_agent.pt"))
                best_reward = np.mean(eval_episode_reward)
                print("Agent saving for reward {}".format(np.mean(eval_episode_reward)))
                nameAgent = "dqn_agent_HL_"+str(history_length + 1)+"_Ep_"+str(i)+"_Reward_"+str(best_reward)+".pt"
                agent.save(os.path.join(model_dir, nameAgent))

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20
    max_timesteps = 500

    env = gym.make('CarRacing-v0').unwrapped
    num_actions = 5
    hist_len = 4
    # Define Q network, target network and DQN agent

    Q = CNN(hist_len).cuda()
    Q_target = CNN(hist_len).cuda()
    agent = DQNAgent(Q, Q_target, num_actions)
    
    train_online(env, agent, max_timesteps=max_timesteps, num_episodes=360, history_length=hist_len, model_dir="./models_carracing")

