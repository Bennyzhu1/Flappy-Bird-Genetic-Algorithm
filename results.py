import numpy as np
import pickle

# Load the saved birds
with open('best_birds_last_gen.pkl', 'rb') as f:
    best_birds_last_gen = pickle.load(f)

# Set up the environment for GUI mode
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

# Run the saved birds in the GUI
for bird in best_birds_last_gen:
    obs, _ = env.reset()
    total_reward = 0
    
    while True:
        action_prob = bird.forward(obs)
        action = 1 if action_prob > 0 else 0
        
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        
        if terminated:
            break

env.close()
