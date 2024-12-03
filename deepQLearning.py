import gymnasium 
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import flappy_bird_gymnasium  

# Initialize Flappy Bird environment
env = gymnasium.make('FlappyBird-v0', render_mode=None, use_lidar=False)

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.99
epsilon = 0.0075
epsilon_decay = 0.9995
epsilon_min = 0.0001
batch_size = 64
memory_size = 50000
target_update_frequency = 10  
num_episodes = 10000

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 128)
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)


state_size = env.observation_space.shape[0]
action_size = env.action_space.n
policy_net = DQN(state_size, action_size) 
target_net = DQN(state_size, action_size) 
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1])  # Random action (exploration)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()  # Best action (exploitation)

def replay():
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Calculate current Q-values
    q_values = policy_net(states).gather(1, actions)

    # Calculate target Q-values
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * discount_factor * max_next_q_values

    # Compute the loss
    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

    # Backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(epsilon):
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        score = 0
        
        while not done:
            action = choose_action(state, epsilon)
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if reward == 1:
                score += 1
            
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            
            replay()

        # Save the trained model
        if score >= 100:
            torch.save(policy_net.state_dict(), "flappybird_dqn_model_100_1.pth")
            break
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Update target network
        if episode % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}, Score: {score}, Total Reward: {total_reward}, Epsilon: {epsilon}")



def play_trained_model():
    env = gymnasium.make('FlappyBird-v0', render_mode=None, use_lidar=False)

    trained_model = DQN(state_size, action_size)

    trained_model.load_state_dict(torch.load("flappybird_dqn_model_100_1.pth"))
    trained_model.eval()  

    total_score = 0
    for episode in range(100):
        state, _ = env.reset()
        done = False
        total_reward = 0
        score = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = trained_model(state_tensor).argmax().item()  
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            if reward == 1:
                score += 1
            
            state = next_state

        print("Episode:", episode, "Total Reward:", total_reward, "Score:", score)
        total_score += score

    print("Finished playing, average score: ", total_score/100)
    env.close()

train(epsilon)
play_trained_model()
