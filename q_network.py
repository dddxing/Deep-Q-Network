from dis import dis
from os import setpgid
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        #--------- YOUR CODE HERE --------------#  [q1, q2, qdot1, qdot2, xgoal, ygoal]
        input_dim = env.observation_space.shape[0]
        print("input dim: ", input_dim)
        self.fc1 = nn.Linear(input_dim, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 15)

        self.bn1d = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax()
        self.env = env
        #---------------------------------------

    def forward(self, x, device):
        #--------- YOUR CODE HERE --------------
        x = torch.Tensor(x).to(device)
        # x = torch.from_numpy(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.softmax(self.fc3(x))
        return x
        #---------------------------------------
        
    def select_discrete_action(self, obs, device):
        # Put the observation through the network to estimate q values for all possible discrete actions
        est_q_vals = self.forward(obs.reshape((1,) + obs.shape), device)
        # Choose the discrete action with the highest estimated q value
        discrete_action = torch.argmax(est_q_vals, dim=1).tolist()[0]
        return discrete_action
                
    def action_discrete_to_continuous(self, discrete_action):
        #--------- YOUR CODE HERE --------------
        # self.env.step(self.q_network.action_discrete_to_continuous(discrete_action))
        # print("discrete_action: ", discrete_action)
        step = 0.1
        dict_ = {
            0: np.array([-step, 0]),
            1: np.array([step, 0]),
            2: np.array([0, -step]),
            3: np.array([0, step]),
            4: np.array([step, step]),
            5: np.array([step, -step]),
            6: np.array([-step, step]),

            7: np.array([0, 0]),

            8: np.array([-2*step, 0]),
            9: np.array([2*step, 0]),
            10: np.array([0, -2*step]),
            11: np.array([0, 2*step]),
            12: np.array([2*step, 2*step]),
            13: np.array([2*step, -2*step]),
            14: np.array([-2*step, 2*step]),
            
            }
        #---------------------------------------
        continous_action = dict_[discrete_action]
        return continous_action

if __name__ == "__main__":
    pass
