import numpy as np
import random
import pickle

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

    def mutate(self, mutation_rate, large_mutation_rate=0.1):
        mutation_mask1 = np.random.randn(*self.weights1.shape) * mutation_rate
        mutation_mask2 = np.random.randn(*self.weights2.shape) * mutation_rate
        
        if np.random.rand() < large_mutation_rate:
            mutation_mask1 *= 10
            mutation_mask2 *= 10
        
        self.weights1 += mutation_mask1
        self.weights2 += mutation_mask2

    def crossover(self, other):
        child = self.clone()
        crossover_point = random.randint(0, self.weights1.shape[1] - 1)
        child.weights1[:, crossover_point:] = other.weights1[:, crossover_point:]
        return child

class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population_size = population_size
        self.networks = [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]

    def evolve(self, fitness_scores, generation, max_generations):
        num_to_select = max(4, int(self.population_size * 0.25))
        top_indices = np.argsort(fitness_scores)[-num_to_select:]
        top_networks = [self.networks[i] for i in top_indices]

        extra_indices = np.random.choice(np.arange(len(fitness_scores)), size=num_to_select, replace=False)
        selected_networks = [self.networks[i] for i in extra_indices]

        mating_pool = top_networks + selected_networks

        new_generation = []
        new_generation.append(top_networks[0])

        mutation_rate = max(0.01, 0.2 * (1 - (generation / max_generations)))

        for _ in range(self.population_size - len(new_generation)):
            parent1, parent2 = random.sample(mating_pool, 2)
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate)
            new_generation.append(child)

        self.networks = new_generation

# Flappy Bird setup
import flappy_bird_gymnasium
import gymnasium

env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)

# Genetic algorithm parameters
population_size = 30
input_size = env.observation_space.shape[0]
hidden_size = 20
output_size = 1
max_generations = 10000

# Initialize the genetic algorithm
genetic_algo = GeneticAlgorithm(population_size, input_size, hidden_size, output_size)

# Run the genetic algorithm across multiple generations
for generation in range(max_generations):
    fitness_scores = []
    game_scores = []  # To store actual game scores (pipes passed)
    
    for network in genetic_algo.networks:
        obs, _ = env.reset()
        total_reward = 0
        survival_time = 0
        pipes_passed = 0  # Actual game score tracking
        
        while True:
            action_prob = network.forward(obs)
            action = 1 if action_prob > 0 else 0
            obs, reward, terminated, _, info = env.step(action)

            survival_time += 1
            total_reward += survival_time * 0.1  # Fitness score based on survival time
            
            if reward == 1:
                pipes_passed += 1  # Increment when a pipe is passed (game score)
                total_reward += 15  # Add to fitness score for passing pipes

            if terminated:
                total_reward -= 5 if survival_time < 50 else 0  # Penalty for early termination
                break
        
        fitness_scores.append(total_reward)
        game_scores.append(pipes_passed)  # Store the actual game score

    # Print the highest game score (pipes passed) for the current generation
    highest_game_score = max(game_scores)
    # print(f"Generation {generation + 1} finished with highest game score: {highest_game_score} pipes passed")

    # Evolve the population based on fitness
    genetic_algo.evolve(fitness_scores, generation, max_generations)

    # Save the best birds after the final generation
    if generation == max_generations - 1:
        best_birds_last_gen = [network.clone() for network in genetic_algo.networks]
        # Save birds to a .txt file
        with open('best_birds_last_gen.txt', 'w') as f:
            for bird in best_birds_last_gen:
                bird_data = {
                    'weights1': bird.weights1.tolist(),
                    'weights2': bird.weights2.tolist()
                }
                f.write(f"{bird_data}\n")

env.close()

# Load and visualize the performance of the best birds
with open('best_birds_last_gen.txt', 'r') as f:
    best_birds_last_gen = []
    for line in f.readlines():
        bird_data = eval(line.strip())  # Read bird weights as a dictionary
        bird = NeuralNetwork(input_size, hidden_size, output_size)
        bird.weights1 = np.array(bird_data['weights1'])
        bird.weights2 = np.array(bird_data['weights2'])
        best_birds_last_gen.append(bird)

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

for bird in best_birds_last_gen:
    obs, _ = env.reset()
    
    while True:
        action_prob = bird.forward(obs)
        action = 1 if action_prob > 0 else 0
        obs, reward, terminated, _, info = env.step(action)
        
        if terminated:
            break

env.close()
