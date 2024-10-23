import numpy as np
import random

# Global variables
# Define parameters for the exponential decay
start_value = 0.5
end_value = 0.01
size = 1000

# Calculate the exponential decay factor
decay_factor = (end_value / start_value) ** (1 / (size - 1))

# Generate the array with exponentially decaying values
exp_decay_array = [start_value * (decay_factor ** i) for i in range(size)]

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

    def mutate(self, mutation_rate):
        mutation_mask1 = np.random.randn(*self.weights1.shape) * mutation_rate
        mutation_mask2 = np.random.randn(*self.weights2.shape) * mutation_rate
        self.weights1 += mutation_mask1
        self.weights2 += mutation_mask2

    def crossover(self, other):
        child = NeuralNetwork(self.weights1.shape[0], self.weights1.shape[1], self.weights2.shape[1])

        # Randomly choose portions of weights1 and weights2 from either parent
        mask1 = np.random.rand(*self.weights1.shape) > 0.5
        mask2 = np.random.rand(*self.weights2.shape) > 0.5

        child.weights1 = np.where(mask1, self.weights1, other.weights1)
        child.weights2 = np.where(mask2, self.weights2, other.weights2)

        return child

class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population_size = population_size
        self.networks = [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]
    
    def evolve(self, fitness_scores, generation):
        top_5_indices = np.argsort(fitness_scores)[-5:]
        top_networks = [self.networks[i] for i in top_5_indices]

        new_generation = []
        
        new_generation.append(top_networks[-1].clone())
        new_generation.append(top_networks[-2].clone()) 

        # Crossover and mutate to fill most of the population
        while len(new_generation) < int(self.population_size * 0.9):
            parent1, parent2 = random.choices(top_networks, k=2)
            child = parent1.crossover(parent2)
            
            # Simulated Annealing: Mutate with a chance and decreasing rate
            if random.random() < 0.95:  # Increased mutation probability
                child.mutate(mutation_rate=exp_decay_array[generation])

            new_generation.append(child)

        # Introduce new random individuals to maintain diversity
        while len(new_generation) < self.population_size:
            new_random = NeuralNetwork(input_size, hidden_size, output_size)
            new_generation.append(new_random)

        # Set new generation
        self.networks = new_generation

#flappy bird
import flappy_bird_gymnasium
import gymnasium

env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)

#genetic algorithm parameters
population_size = 30
input_size = env.observation_space.shape[0]
hidden_size = 8
output_size = 1

#initial population
genetic_algo = GeneticAlgorithm(population_size, input_size, hidden_size, output_size)

best_score = -1

#run through the game for each neural network in the population
for generation in range(size):  # number of generations
    fitness_scores = []
    print(f"Generation {generation}")
    
    for network in genetic_algo.networks:
        score = 0
        obs, _ = env.reset()
        total_reward = 0
        
        # Maybe manually give a reward where, if they die, see how close they are to the next pipes opening hole, and give a reward based on that
        while True:
            action_prob = network.forward(obs)
            action = 1 if action_prob > 0 else 0
            
            obs, reward, terminated, _, info = env.step(action)
            total_reward += reward
            if reward == 1:
                score += 1
            if terminated:
                break
        if score > best_score:
            best_score = score
    
        fitness_scores.append(total_reward)
    
    #create new generation of birds
    genetic_algo.evolve(fitness_scores, generation)
    print(f"Best score in generation {generation}: {best_score}")

print(f"Best score: {best_score}")
env.close()
