import argparse
from threading import main_thread
import numpy as np
import time
import random
import os
import signal
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
from q_network import QNetwork
from arm_env import ArmEnv
import copy 

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

EPISODES = 300 
EXPLORE_EPI_END = int(0.1*EPISODES) # initial exploration when agent will explore and no training
TEST_EPI_START = int(0.9*EPISODES ) # agent will be tested from this episode
EPS_START = 1.0 # e-greedy threshold start value
EPS_END = 0.05 # e-greedy threshold end value
EPS_DECAY = 1+np.log(EPS_END)/(0.6*EPISODES) # e-greedy threshold decay
GAMMA = 0.999 # Q-learning discount factor
LR = 0.001 # NN optimizer learning rate
MINIBATCH_SIZE = 32 # Q-learning batch size
ITERATIONS = 10 # Number of iterations for training
REP_MEM_SIZE = 10000 # Replay Memory size

rewards_collection = []
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

        self.optimizer = optim.Adam(self.q_network.parameters(), LR)
        self.lossCriterion = torch.nn.MSELoss()
        self.memory = ReplayBuffer(REP_MEM_SIZE) # Instantiate the Replay Memory for storing agent’s experiences # Initialize internal variables
        self.steps_done = 0
        self.episode_durations = []
        self.avg_episode_duration = []
        self.epsilon = EPS_START
        self.epsilon_history = []
        self.mode = ""


    def save_model(self, episode_num, episode_reward, args):
        model_folder_name = f'episode_{episode_num:06d}_reward_{round(episode_reward):03d}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(self.q_network.state_dict(), os.path.join(args.save_dir, model_folder_name, 'q_network.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "q_network.pth")}\n')

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return np.random.randint(0, 6)
        else:
            goal = self.env.goal
            obs = np.concatenate((state, goal), axis=None, dtype=np.float32)
            
            discrete_action = self.q_network.select_discrete_action(state, self.device) # i think should be obs, but 8
            return discrete_action
    
    def run_episode(self, e, env):
        state = self.env.reset()
        done = False
        steps = 0
        if e < EXPLORE_EPI_END:
            self.epsilon = EPS_START
            self.mode = "Exploring"
            print(self.mode)

        elif EXPLORE_EPI_END <= e <= TEST_EPI_START:
            self.epsilon = self.epsilon*EPS_DECAY
            self.mode = "Training"
            print(self.mode)

        elif e > TEST_EPI_START:
            self.epsilon = 0.0
            self.mode = "Testing"
            print(self.mode)
        self.epsilon_history.append(self.epsilon)

        while True: # Iterate until episode ends (i.e. a terminal state is reached)
            # print(state.shape)
            discrete_action = self.select_action(torch.FloatTensor(state)) # Select action based on epsilon-greedy approach

            continuous_action = self.q_network.action_discrete_to_continuous(discrete_action)
            # Get next state and reward from environment based on current action
            next_state, reward, done, _ = env.step(continuous_action)

            self.memory.put((state, discrete_action, reward, next_state, done))

            if EXPLORE_EPI_END <= e <= TEST_EPI_START:
                self.learn()
                state = next_state
                steps += 1

            if done: # Print information after every episode
                print(f"Episode = {e} | reward = {reward} | epsilon {self.epsilon}")
                self.episode_durations.append(steps)
                return reward


    def learn(self):

        if len(self.memory) < MINIBATCH_SIZE:
            return

        for i in range(ITERATIONS):
            
            batch_state, batch_action, batch_reward, batch_next_state, done = self.memory.sample(MINIBATCH_SIZE)
            batch_state = torch.FloatTensor(batch_state)
            batch_action = torch.FloatTensor(batch_action)
            batch_reward = torch.FloatTensor(batch_reward)
            batch_next_state = torch.FloatTensor(batch_next_state)
            done = torch.FloatTensor(done)

            current_q_values = self.q_network(batch_state, self.device).gather(1, torch.FloatTensor(batch_action).type(torch.int64).unsqueeze(1))
            max_next_q_values = self.q_network(batch_next_state, self.device).detach().max(1)[0]
            expected_q_values = batch_reward + (GAMMA * max_next_q_values)

            loss = self.lossCriterion(current_q_values, expected_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, args):
        #--------- YOUR CODE HERE --------------
        for e in range(EPISODES):
            episode_reward = self.run_episode(e, self.env)
            self.save_model(e, episode_reward, args)
        #---------------------------------------        


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
    # tdqn.train(args)