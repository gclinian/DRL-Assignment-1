import random
import numpy as np
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv


def update_passenger_picked(state, prev_action, passenger_picked):
    taxi_pos = (state[0], state[1])
    stations = [(state[2 * i + 2], state[2 * i + 3]) for i in range(4)]
    passenger_look = state[14]
    on_station = False
    for station in stations:
        if taxi_pos == station:
            on_station = True
            break

    if not passenger_picked and passenger_look == 1 and on_station and prev_action == 4:
        return True
        
    if passenger_picked and prev_action == 5:
        return False
    
    return passenger_picked

def get_state_extend(state, passenger_picked):
    taxi_pos = (state[0], state[1])
    stations = [(state[2 * i + 2], state[2 * i + 3]) for i in range(4)]

    def trans(x, y):
        if x < y:
            return 1
        elif x > y:
            return -1
        else:
            return 0

    dirs = [(trans(taxi_pos[0], stations[i][0]), trans(taxi_pos[1], stations[i][1])) for i in range(4)]

    return (state[10], state[11], state[12], state[13])



def get_action(state_extend, q_table, epsilon, n_actions):
    """Choose an action using an epsilon-greedy policy."""
    # Initialize state in Q-table if unseen
    if state_extend not in q_table:
        q_table[state_extend] = np.zeros(n_actions)
        q_table[state_extend][4] = -10000000
        q_table[state_extend][5] = -10000000
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return int(np.argmax(q_table[state_extend]))

def main():
    # Create the environment
    env = SimpleTaxiEnv(grid_size=10, fuel_limit=5000)
    n_actions = 6  # [Move Down, Move Up, Move Right, Move Left, PICKUP, DROPOFF]
    
    # Initialize Q-table as a dictionary: key=state (tuple), value=np.array of Q-values
    q_table = {}
    
    # Hyperparameters for Q-learning
    alpha = 0.5     # Learning rate
    gamma = 0.995       # Discount factor
    epsilon = 1       # Initial exploration rate
    epsilon_min = 0.1    # Minimum exploration rate
    epsilon_decay = 0.9995  # Decay factor for exploration rate
    num_episodes = 10000  # Number of training episodes
    max_steps = 5000    # Maximum steps per episode

    for episode in range(num_episodes):
        # size_limit = 5 + episode // 1000
        # grid_size = random.randint(5, size_limit)
        # env = SimpleTaxiEnv(grid_size=10, fuel_limit=5000)
        state, _ = env.reset()
        passenger_picked = False
        prev_passenger_picked = False
        prev_action = None
        state_extend = get_state_extend(state, passenger_picked)
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Choose an action using epsilon-greedy
            action = get_action(state_extend, q_table, epsilon, n_actions)
            next_state, reward, done, _ = env.step(action)
            
            prev_action = action
            prev_passenger_picked = passenger_picked
            passenger_picked = update_passenger_picked(state, prev_action, passenger_picked)
            next_state_extend = get_state_extend(next_state, passenger_picked)
            # Ensure the next state is in the Q-table
            
            if next_state_extend not in q_table:
                q_table[next_state_extend] = np.zeros(n_actions)
                q_table[state_extend][4] = -10000000
                q_table[state_extend][5] = -10000000
            
            # reward shaping
            

            total_reward += reward
            # Update Q-value using the Q-learning update rule
            q_table[state_extend][action] += alpha * (reward + gamma * np.max(q_table[next_state_extend]) - q_table[state_extend][action])
            state = next_state
            state_extend = next_state_extend
            step += 1

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_reward} | Episilon: {epsilon}")

    # Save the Q-table using pickle
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Training completed. Q-table saved to 'q_table.pkl'.")

if __name__ == "__main__":
    main()
