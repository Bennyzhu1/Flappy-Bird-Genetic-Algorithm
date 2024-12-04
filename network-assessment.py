import numpy as np
import pickle
import gymnasium # type: ignore
from geneticAlgorithm import NeuralNetwork

def assess_trained_networks(pickle_filename, runs_per_network=10):
    with open(pickle_filename, "rb") as f:
        loaded_networks = pickle.load(f)
    
    env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    
    total_scores = []
    total_fitness = []
    scores_ranked = []

    generation = 0
    for network in loaded_networks:
        print(f'network {network}')
        scores = []
        fitness_values = []
        
        for _ in range(runs_per_network):
            if _ % 1 == 0:
                print(f'run {_}')
            obs, _ = env.reset()
            score = 0
            total_reward = 0
            
            while True:
                action_prob = network.forward(obs)
                action = 1 if action_prob > 0 else 0
                
                obs, reward, terminated, _, info = env.step(action)
                total_reward += reward
                
                # when pipe passed reward = 1
                if reward == 1:
                    score += 1
                    
                if terminated:
                    break
            
            scores.append(score)
            fitness_values.append(total_reward)
        
        avg_score = np.mean(scores)
        avg_fitness = np.mean(fitness_values)

        scores_ranked.append((avg_score, generation))
        
        total_scores.append(avg_score)
        total_fitness.append(avg_fitness)
        generation += 1
    
    overall_avg_score = np.mean(total_scores)
    overall_avg_fitness = np.mean(total_fitness)
    
    env.close()

    scores_ranked.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Overall average score: {overall_avg_score}")
    print(f"Overall average fitness: {overall_avg_fitness}")
    print(f"Scores ranked: {scores_ranked}")
    return overall_avg_score, overall_avg_fitness

if __name__ == "__main__":
    assess_trained_networks("./Flappy-Networks/BestNeural-30947Score.pkl", runs_per_network=10)
