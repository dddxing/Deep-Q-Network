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
        
        self.step = 0.2
        self.action_space = {
            
            0: np.array([-0.5 * self.step, -1  * self.step]),
            1: np.array([-0.5 * self.step, -0.5 * self.step]),
            2: np.array([-0.5 * self.step,  0  * self.step]),
            3: np.array([-0.5 * self.step, 0.5 * self.step]),
            4: np.array([-0.5 * self.step,  1  * self.step]),

            5: np.array([-2 * self.step, -1  * self.step]),
            6: np.array([-2 * self.step, -0.5 * self.step]),
            7: np.array([-2 * self.step,  0  * self.step]),
            8: np.array([-2 * self.step, 0.5 * self.step]),
            9: np.array([-2 * self.step,  1  * self.step]),

            10: np.array([-1 * self.step, -1  * self.step]),
            11: np.array([-1 * self.step, -0.5 * self.step]),
            12: np.array([-1 * self.step,  0  * self.step]),
            13: np.array([-1 * self.step, 0.5 * self.step]),
            14: np.array([-1 * self.step,  1  * self.step]),

            15: np.array([0 * self.step, -1  * self.step]),
            16: np.array([0 * self.step, -0.5 * self.step]),
            17: np.array([0 * self.step,  0  * self.step]),
            18: np.array([0 * self.step, 0.5 * self.step]),
            19: np.array([0 * self.step,  1  * self.step]),

            20: np.array([1 * self.step, -1  * self.step]),
            21: np.array([1 * self.step, -0.5 * self.step]),
            22: np.array([1 * self.step,  0  * self.step]),
            23: np.array([1 * self.step, 0.5 * self.step]),
            24: np.array([1 * self.step,  1  * self.step]),

            25: np.array([0.5 * self.step, -1  * self.step]),
            26: np.array([0.5 * self.step, -0.5 * self.step]),
            27: np.array([0.5 * self.step,  0  * self.step]),
            28: np.array([0.5 * self.step, 0.5 * self.step]),
            29: np.array([0.5 * self.step,  1  * self.step]),

            30: np.array([-0.5 * self.step,  2 * self.step]),
            31: np.array([-2 * self.step, -2 * self.step]),
            32: np.array([-2 * self.step,  2 * self.step]),
            33: np.array([-1 * self.step,  2 * self.step]),
            34: np.array([-1 * self.step, -2 * self.step]),
            35: np.array([0 * self.step, -2 * self.step]),
            36: np.array([0 * self.step,  2 * self.step]),
            37: np.array([1 * self.step, -2 * self.step]),
            38: np.array([-0.5 * self.step, -2 * self.step]),
            39: np.array([1 * self.step,  2  * self.step]),
            40: np.array([2 * self.step, -2  * self.step]),
            41: np.array([2 * self.step, -1  * self.step]),
            42: np.array([2 * self.step, -0.5 * self.step]),
            43: np.array([2 * self.step,  0  * self.step]),
            44: np.array([2 * self.step, 0.5 * self.step]),
            45: np.array([2 * self.step,  1  * self.step]),
            46: np.array([2 * self.step,  2  * self.step]),
            47: np.array([0.5 * self.step, -2  * self.step]),
            48: np.array([0.5 * self.step,  2  * self.step]),
            }
        self.action_space_dim = len(self.action_space)
        input_dim = env.observation_space.shape[0]
        output_dim = len(self.action_space)
        print("input dim: ", input_dim)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            
            nn.ReLU(),
            nn.Linear(128, 64),
            
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, output_dim),
            # nn.Tanh()
        )
        self.env = env

        #---------------------------------------

    def forward(self, x, device):
        #--------- YOUR CODE HERE --------------
        
        x = torch.Tensor(x).to(device)
        x = self.layers(x)
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
        #---------------------------------------
        continous_action = self.action_space[discrete_action]
        return continous_action

if __name__ == "__main__":
    pass