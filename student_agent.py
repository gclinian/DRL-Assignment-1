
# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

# Load the pre-trained Q-table
with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)


prev_action = None
passenger_picked = False

def get_state(obs):
    # decode obs
    taxi_pos = (obs[0], obs[1])
    stations = [(0,0) for i in range(4)]
    stations[0] = (obs[2], obs[3])
    stations[1] = (obs[4], obs[5])
    stations[2] = (obs[6], obs[7])
    stations[3] = (obs[8], obs[9])
    
    passenger_look = obs[14]
    

    global prev_action
    global passenger_picked

    on_station = False
    for station in stations:
        if taxi_pos == station:
            on_station = True
            break

    if not passenger_picked and passenger_look == 1 and on_station and prev_action == 4:
        passenger_picked = True
        
    if passenger_picked and prev_action == 5:
        passenger_picked = False
    
    state = (*obs, passenger_picked)
    return state
        

         


def get_action(obs):
    state_extend = get_state(obs)
    if state_extend in Q_table:     
        action = np.argmax(Q_table[state_extend])
    else:
        action = np.random.choice([0,1,2,3])
    prev_action = action
    return action
    













'''
import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import gym


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # A simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)


def state_to_tensor(state):
    # State is a 16-element tuple; turn it into a float tensor of shape [1, state_dim].
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


# Load model
loaded_dqn = DQN(state_dim=16, action_dim=6)
loaded_dqn.load_state_dict(torch.load("dqn_taxi_model.pth"))
loaded_dqn.eval()  # set to evaluation mode


def get_action(state):
    """
    Given a trained DQN model and a state (tuple), return the action (int).
    """
    with torch.no_grad():
        q_values = loaded_dqn(state_to_tensor(state))
        action = int(torch.argmax(q_values, dim=1).item())
    return action
'''