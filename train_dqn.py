import argparse

import numpy as np
import time
import random
import os
import signal
from contextlib import contextmanager
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from replay_buffer import ReplayBuffer
from q_network import QNetwork
from arm_env import ArmEnv


#---------- Utils for setting time constraints -----#
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
#---------- End of timing utils -----------#

reward_collection = []
loss_collection = []
eps_collection = []
loss_collection = []

class TrainDQN:

    @staticmethod
    def add_arguments(parser):
        # Common arguments
        parser.add_argument('--learning_rate', type=float, default=7e-4,
                            help='the learning rate of the optimizer')
        # LEAVE AS DEFAULT THE SEED YOU WANT TO BE GRADED WITH
        parser.add_argument('--seed', type=int, default=1,
                            help='seed of the experiment')
        parser.add_argument('--save_dir', type=str, default='models',
                            help="the root folder for saving the checkpoints")
        parser.add_argument('--gui', action='store_true', default=False,
                            help="whether to turn on GUI or not")
        # 7 minutes by default
        parser.add_argument('--time_limit', type=int, default=7*60,
                            help='time limits for running the training in seconds')

    def __init__(self, env, device, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        self.env = env
        self.env.seed(args.seed)
        self.env.observation_space.seed(args.seed)
        self.device = device
        self.q_network = QNetwork(env).to(self.device)
        self.t_network = QNetwork(env).to(self.device)
        print(self.device.__repr__())
        print(self.q_network)

    def save_model(self, episode_num, episode_reward, args):
        model_folder_name = f'episode_{episode_num:06d}_reward_{round(episode_reward):03d}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(self.q_network.state_dict(), os.path.join(args.save_dir, model_folder_name, 'q_network.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "q_network.pth")}\n')
                
    def train(self, args):

        #--------- YOUR CODE HERE --------------
        
        
        lr = 0.0001 #args.learning_rate0.007
        optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        loss_fcn = nn.MSELoss()
        N = 10000
        replay_buffer = ReplayBuffer(N)
        

        # self.t_network.load_state_dict(self.q_network.state_dict())
        # self.t_network.eval()        

        gamma = 0.9
        num_episodes = 10000
        batch_size = 128
        epsilon = 0.015
        num_steps = 500
        update_steps = 5

        eps_start = 1
        eps_end = 0.05
        eps_decay = 1000

        global_steps = 0

        for i_episode in range(1, num_episodes+1):
            
            state  = self.env.reset() 
            episode_loss = 0
            episode_reward = 0
            # print("episode, ", i_episode)
            
            
            for step in range(1, num_steps+1):
                # With probability ε, at = random

                epsilon = eps_end + (eps_start - eps_end) * np.exp(-1. * global_steps / eps_decay)
                

                if np.random.random() < epsilon:
                    # print("Exploring..")
                    discrete_action = np.random.randint(0, self.q_network.action_space_dim-1)
                    
                
                # otherwise at = maxaQA(s, a)
                else:

                    discrete_action = self.q_network.select_discrete_action(torch.from_numpy(state).float(), self.device)

                # discrete to continuous
                continuous_action = self.q_network.action_discrete_to_continuous(discrete_action)
                
                # Execute action
                next_state, reward, done, _ = self.env.step(continuous_action)
                
                #if done is terminal: break 
                if done or next_state is None:
                    break                
                
                # update reward 
                episode_reward += reward
                
                # store transition into replay memory
                replay_buffer.put((state, discrete_action, reward, next_state, done)) # not sure if action should be disc or cont?
                
                # train network 
                # accumulate some steps to a batch
                if step > batch_size:

                    # batches of stuff (state, action, reward, next_state, done_mark)
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                    states = torch.FloatTensor(states)
                    actions = torch.FloatTensor(actions)
                    rewards = torch.FloatTensor(rewards)
                    next_states = torch.FloatTensor(next_states)
                    dones = torch.FloatTensor(dones)


                    # current_q
                    max_next_q_values = self.t_network(next_states, self.device).max(1)[0].detach()
                    expected_q_values = rewards.to(self.device) + (gamma * max_next_q_values * (1 - dones.to(self.device)))

                    current_q_values = self.q_network(states, self.device).gather(1, actions.view(-1, 1).type(torch.int64).to(self.device)).squeeze()

                    loss = loss_fcn(current_q_values.squeeze(), expected_q_values.squeeze())
                    episode_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                if step % update_steps == 0:
                    # print("set 2 nn equal")
                    self.t_network.load_state_dict(self.q_network.state_dict())
                
                state = next_state    

            reward_collection.append(episode_reward)
            eps_collection.append(epsilon)
            loss_collection.append(episode_loss)
            global_steps += step

            print(f"episode: {i_episode}, global_steps: {global_steps}, episode_reward: {episode_reward}, loss {episode_loss}, eps: {epsilon}")
            self.save_model(i_episode, episode_reward, args)
            #---------------------------------------        

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TrainDQN.add_arguments(parser)
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    if not args.seed: args.seed = int(time.time())

    env = ArmEnv(args)
    device = torch.device('cpu')
    # declare Q function network and set seed
    tdqn = TrainDQN(env, device, args)
    # run training under time limit
    try:
        with time_limit(args.time_limit):
            tdqn.train(args)
    except TimeoutException as e:
        print("You ran out of time and your training is stopped!")

    mv_avg_r = moving_average(reward_collection, 20)
    mv_avg_l = moving_average(loss_collection, 20)
    fig, ax = plt.subplots(2)


    ax[0].plot(reward_collection, "r--", label="episode rewards")
    ax[0].plot(mv_avg_r, "b-", label="moving average reward")
    ax[1].plot(eps_collection, label="epison")
    ax[1].plot(loss_collection, label="episode loss") 
    
    ax[0].set_xlabel("Episode")
    ax[1].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")
    ax[1].set_ylabel("Loss")
    ax[0].legend()
    ax[1].legend()
    
    plt.show()
    