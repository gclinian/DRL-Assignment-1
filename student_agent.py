# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

# Load the pre-trained Q-table
with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)


def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    # Selects the best action using the trained Q-table.
    if obs in Q_table:     
        return np.argmax(Q_table[obs])
    else:
        return np.random.choice([0,1,2,3])
    # return np.random.choice([0,1,2,3,4,5])
    # You can submit this random agent to evaluate the performance of a purely random strategy.

