
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
    stations = [(obs[2 * i + 2], obs[2 * i + 3]) for i in range(4)]

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


    def trans(x, y):
        if x < y:
            return 1
        elif x > y:
            return -1
        else:
            return 0


    dirs = [(trans(taxi_pos[0], stations[i][0]), trans(taxi_pos[1], stations[i][1])) for i in range(4)]
    
    return (*dirs[0], *dirs[1], *dirs[2], *dirs[3], obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], passenger_picked)
    
        

         


def get_action(obs):
    state_extend = get_state(obs)
    if False and state_extend in Q_table:     
        action = np.argmax(Q_table[state_extend])
    else:
        action = np.random.choice([0,1,2,3])
    prev_action = action
    return action
