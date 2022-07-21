import tensorflow as tf
import torch
import numpy as np
from reinforcement_learning.agent.replay_buffer import ReplayBuffer
import random

def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4, history_length=0):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions


    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        #
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets and loss 
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)
        self.Q.train()
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        if self.replay_buffer.__len__() >= self.batch_size:
            states, actions, next_states, rewards, dones = self.replay_buffer.next_batch(self.batch_size)
            Q_targets_next = self.Q_target(next_states.to(torch.float32))
            Q_targets = (rewards + (self.gamma * torch.max(Q_targets_next, 1).values * ~dones)).to(torch.float32).unsqueeze(1)
            Q_expected = torch.gather(self.Q(states.to(torch.float32)), 1, actions.unsqueeze(1))

            #Compute loss()
            loss = self.loss_function(Q_expected, Q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """

        self.Q.eval()
        state = torch.from_numpy(state.astype(np.float32)).cuda()
        action_values = self.Q(state)

        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # take greedy action (argmax)
            action_id = np.argmax(action_values.cpu().detach().numpy())
        else:
            # sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            if self.num_actions == 2:
                action_id = np.random.choice(np.arange(self.num_actions))
            else:
                action_id = np.random.choice(np.arange(self.num_actions), p=[0.5, 0.2, 0.2, 0.05, 0.05])
          
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
