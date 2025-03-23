
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


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        # A simple 2-hidden-layer MLP for demonstration
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



policy_net = QNetwork(state_size=16, action_size=6)
policy_net.load_state_dict(torch.load("taxi_dqn_model.pth"))
policy_net.eval()



def get_action(state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_values = policy_net(state_tensor)
    action = action_values.argmax().item()
    return action

'''