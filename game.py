import numpy as np
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
    
    def forward(self, x):
        hidden = np.dot(x, self.weights1)
        hidden = np.tanh(hidden)
        output = np.dot(hidden, self.weights2)
        return output

    def clone(self):
        clone = NeuralNetwork(self.weights1.shape[0], self.weights1.shape[1], self.weights2.shape[1])
        clone.weights1 = np.copy(self.weights1)
        clone.weights2 = np.copy(self.weights2)
        return clone

    def mutate(self, mutation_rate=0.05):
        mutation_mask1 = np.random.randn(*self.weights1.shape) * mutation_rate
        mutation_mask2 = np.random.randn(*self.weights2.shape) * mutation_rate
        self.weights1 += mutation_mask1
        self.weights2 += mutation_mask2

class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population_size = population_size
        self.networks = [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]
    
    def evolve(self, fitness_scores):
        #select top 2 fittest birds
        top_2_indices = np.argsort(fitness_scores)[-2:]
        top_networks = [self.networks[i] for i in top_2_indices]

        #create the new generation
        new_generation = []
        for top_network in top_networks:
            #add top two birds to next gen
            new_generation.append(top_network)
            #clone the top birds 4 times and mutate the clones
            for _ in range(4):
                cloned_network = top_network.clone()
                cloned_network.mutate(mutation_rate=0.05)
                new_generation.append(cloned_network)
        
        #set new generation
        self.networks = new_generation

#flappy bird
import flappy_bird_gymnasium
import gymnasium

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

#genetic algorithm parameters
population_size = 10
input_size = env.observation_space.shape[0]
hidden_size = 8
output_size = 1

#initial population
genetic_algo = GeneticAlgorithm(population_size, input_size, hidden_size, output_size)

#run through the game for each neural network in the population
for generation in range(100):  # number of generations
    fitness_scores = []
    
    for network in genetic_algo.networks:
        obs, _ = env.reset()
        total_reward = 0
        
        while True:
            action_prob = network.forward(obs)
            action = 1 if action_prob > 0 else 0
            
            obs, reward, terminated, _, info = env.step(action)
            total_reward += reward
            
            if terminated:
                break
        
        fitness_scores.append(total_reward)
    
    #create new generation of birds
    genetic_algo.evolve(fitness_scores)

env.close()
