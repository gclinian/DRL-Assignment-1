import random
import numpy as np
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv


# state design
def get_state(obs):
    # decode obs
    taxi_pos = (obs[0], obs[1])
    stations = [(0,0) for i in range(4)]
    stations[0] = (obs[2], obs[3])
    stations[1] = (obs[4], obs[5])
    stations[2] = (obs[6], obs[7])
    stations[3] = (obs[8], obs[9])
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east = obs[12]
    obstacle_west = obs[13]
    passenger_look = obs[14]
    destination_look = obs[15]

    on_station = False
    for station in stations:
        if taxi_pos == station:
            on_station = True
            break

    if passenger_look == 1 and on_station:
        pass


def get_action(state, q_table, epsilon, n_actions):
    """Choose an action using an epsilon-greedy policy."""
    # Initialize state in Q-table if unseen
    if state not in q_table:
        q_table[state] = np.zeros(n_actions)
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return int(np.argmax(q_table[state]))

def main():
    # Create the environment
    env = SimpleTaxiEnv(grid_size=10, fuel_limit=500)
    n_actions = 6  # [Move Down, Move Up, Move Right, Move Left, PICKUP, DROPOFF]
    
    # Initialize Q-table as a dictionary: key=state (tuple), value=np.array of Q-values
    q_table = {}
    
    # Hyperparameters for Q-learning
    alpha = 0.001         # Learning rate
    gamma = 0.995       # Discount factor
    epsilon = 1.0        # Initial exploration rate
    epsilon_min = 0.1    # Minimum exploration rate
    epsilon_decay = 0.99997  # Decay factor for exploration rate
    num_episodes = 70000  # Number of training episodes
    max_steps = 500    # Maximum steps per episode

    for episode in range(num_episodes):
        # size_limit = 5 + episode // 1000
        # grid_size = random.randint(5, size_limit)
        # env = SimpleTaxiEnv(grid_size=10, fuel_limit=5000)
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            # Choose an action using epsilon-greedy
            action = get_action(state, q_table, epsilon, n_actions)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Ensure the next state is in the Q-table
            if next_state not in q_table:
                q_table[next_state] = np.zeros(n_actions)
            
            # Update Q-value using the Q-learning update rule
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            state = next_state
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
