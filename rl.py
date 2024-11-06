import gymnasium 
import numpy as np
import random
from collections import defaultdict
import flappy_bird_gymnasium  

# Initialize the Flappy Bird environment
env = gymnasium.make('FlappyBird-v0', render_mode=None, use_lidar=False)

# Hyperparameters
learning_rate = 0.3           # Alpha
discount_factor = 0.9        # Gamma
epsilon = 1.0                 # Initial epsilon for epsilon-greedy policy
epsilon_decay = 0.99999         # Decay rate for epsilon
epsilon_min = 0.01            # Minimum epsilon value
num_episodes = 100000           # Number of episodes to train

# Initialize Q-table with default value 0 for unseen states
Q_table = defaultdict(lambda: np.zeros(env.action_space.n))

# Function to discretize continuous states for the Q-table
def discretize_state(state):
    """Convert continuous state to a discrete tuple of bin indices."""

    # Return a tuple of discretized features
    return (int(state[0] // 20), int(state[1] // 20), int(state[2] // 20), int(state[3] // 20),
            int(state[4] // 20), int(state[5] // 20), int(state[6] // 20), int(state[7] // 20),
            int(state[8] // 20), int(state[9] // 20), int(state[10] // 20), int(state[11] // 20))


# Q-learning algorithm
for episode in range(num_episodes):
    state, _ = env.reset()
    state = discretize_state(state)  # Discretize initial state
    done = False
    total_reward = 0
    
    while not done:
        # print('state: ', state)
        score = 0
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0,1)  # Explore
        else:
            action = np.argmax(Q_table[state])  # Exploit best-known action

        # Take action, observe next state and reward
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)  # Discretize next state
        total_reward += reward
        if reward == 1:
            score += 1
        
        # Update Q-value for current state-action pair
        best_next_action = np.argmax(Q_table[next_state])
        td_target = reward + discount_factor * Q_table[next_state][best_next_action]
        Q_table[state][action] += learning_rate * (td_target - Q_table[state][action])
        
        # Move to the next state
        state = next_state  

    # Decay epsilon after each episode
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode + 1}, Score: {score}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()
