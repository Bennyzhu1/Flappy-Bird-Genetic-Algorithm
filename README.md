# Flappy Bird AI: Genetic Algorithm and Deep Q-Learning

"*To flap or not to flap*"

## Overview

This project explores two different artificial intelligence approaches to playing Flappy Bird: a genetic algorithm and deep Q-learning.  The goal is to train agents that can successfully navigate the simple yet difficult game of Flappy Bird and achieve high scores than a person usually is able to.  

## Features

- **Genetic Algorithm**
    - Evolves a population of neural networks over multiple generations.
    - Employs crossover and mutation to optimize network weights.
    - Uses an exponential decay for the mutation rate, inspired by simulated annealing.
    - Periodically saves the best-performing networks.

- **Deep Q-Learning**
    - Implements a Deep Q-Network (DQN) to learn an optimal action policy.
    - Uses experience replay and a target network for improved stability.
    - Employs an epsilon-greedy strategy for a working balance between exploration and exploitation.
    - Saves the trained DQN model upon reaching a score of 100.

## Document Overview

- `geneticAlgorithm.py`: Implements the genetic algorithm approach.
- `deepQLearning.py`: Implements the deep Q-learning approach.
- `network-assessment.py`: Provides a script to evaluate the performance of saved networks. Currently preset to one of our genetic algorithm results.
- `Flappy-Networks/`: Directory containing saved `.pth` (DQN model) and `.pkl` (genetic algorithm networks) files.
- `requirements.txt`: Project dependencies.
- `Makefile`:  Facilitates running the pre-trained models (see Getting Started).

## Getting Started

### Prerequisites

- Python 3.9+
- Please make sure you have the dependencies listed in `requirements.txt` installed.  You can install them using:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Models

We've included a Makefile to simplify running the pre-trained models, to run our project:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Flappy-Bird-Genetic-Algorithm.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Flappy-Bird-Genetic-Algorithm
   ```

3. Run whichever model you want to see:
   
   Genetic:
   ```bash
   make genetic  # Runs the pre-trained genetic algorithm
   OR
   python3 geneticAlgorithm.py
   ```

   DeepQN
   ```bash
   make dqn      # Runs the pre-trained Q-learning model
   OR
   python3 deepQLearning.py
   ```

## Model and Training Details

### Genetic Algorithm

The genetic algorithm trains a population of neural networks.  The best-performing networks are selected for reproduction through crossover and mutation. The mutation rate decreases over generations using an exponential decay schedule.  The saved `.pkl` files in the `Flappy-Networks/` directory contain the weights of these evolved networks.

### Deep Q-Learning

The deep Q-learning agent uses a DQN to approximate the optimal action-value function. The agent is trained using experience replay and a target network. The epsilon-greedy exploration strategy balances exploration and exploitation. The saved `.pth` file in the `Flappy-Networks/` directory represents the trained DQN model.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request.

## Acknowledgments

This project utilizes the `flappy-bird-gymnasium` environment, for which we express our gratitude to the developers. You can find the link to their project in the resources below.

## Resources
This project leverages `gymnasium`: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

Website for Flappybirdgym the environment we used: [https://pypi.org/project/flappy-bird-gymnasium/](https://pypi.org/project/flappy-bird-gymnasium/)
