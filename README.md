# Flappy Bird AI

## Overview
Welcome to the Flappy Bird AI repository! This project is a machine learning experiment that uses the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to train a neural network to play the game Flappy Bird. The game is implemented in Python using the Pygame library.

## Project Architecture and Hierarchy
The project is structured as follows:
- `Games`: Contains the main game executable files.
- `Objects`: Includes the classes for game objects like the base, bird, and pipes.
- `imgs`: Holds all the image assets used in the game.
- `game_stats`: txt file which contains information about played games by ai
- `README.md`: The file you are reading now, which provides documentation for the project.

### Key Components:
- `Flappy_bird_OOP.py`: The main game class file using Object-Oriented Principles.
- `config.py`: Contains game settings.
- `neat_config.txt`: Contains NEAT algorithm settings.
- `functions.py`: Helper functions used across the game.
- `main.py`: The entry point of the game to start playing.

## Features
- Utilizes the NEAT algorithm for evolving neural networks to play the game autonomously.
- Implements a Pygame-based environment for visual representation of gameplay.
- Tracks and displays real-time statistics such as score and fitness levels.

## NEAT Algorithm Inputs
The NEAT (NeuroEvolution of Augmenting Topologies) algorithm is used to control the bird's movements in the game. The inputs to the neural network determine the bird's actions at each frame. The following inputs are provided to the NEAT algorithm:

1. `bird.y`: The bird's current y-coordinate.
2. `abs(bird.y - self.pipes[pipe_ind].height)`: The absolute difference between the bird's y-coordinate and the top pipe's height.
3. `abs(bird.y - self.pipes[pipe_ind].bottom)`: The absolute difference between the bird's y-coordinate and the bottom pipe's position.
4. `bird.vel`: The current vertical velocity of the bird.
5. `abs(bird.x - self.pipes[pipe_ind].x)`: The absolute difference between the bird's x-coordinate and the pipe's x-coordinate.

These inputs are processed by the neural network to output the decision of whether the bird should jump or not at each time step. The goal is for the AI to learn the optimal timing of jumps to navigate through the pipes without colliding.

### Understanding the Inputs:
- The `bird.y` input allows the neural network to consider the bird's vertical position.
- The vertical distances to the next pipes (`height` and `bottom`) help the AI gauge the gaps it needs to pass through.
- The `bird.vel` input is crucial for understanding the bird's current motion.
- The horizontal distance to the next pipe (`bird.x - pipe.x`) helps the AI predict when to jump to clear the pipe.

By providing these specific inputs, the NEAT algorithm can evolve neural networks that effectively play Flappy Bird over successive generations.


# Analysis

In this project, the NEAT (NeuroEvolution of Augmenting Topologies) algorithm is configured with several parameters to optimize the neural network that controls the bird's movements. The following statistics and configuration parameters are tracked and adjusted during the simulation:

- **Game Index**: Identifier for the game instance.
- **Generation**: The current generation number in the evolutionary algorithm.
- **Max Score**: The highest score achieved by any bird in the current generation.
- **Max Fitness**: The highest fitness value achieved by any bird in the current generation.
- **Population Size**: The total number of individual birds (networks) in the current population.
- **Fitness Threshold**: The predetermined fitness score at which the simulation is considered to have found a solution.
- **Activation Functions**: Specifies the type of activation function used in the neural network (e.g., sigmoid, tanh).
- **Compatibility Disjoint Coefficient**: A coefficient that measures species diversity within the population, influencing speciation.
- **Connection Add Probability**: The likelihood of a mutation that adds a new connection between nodes.
- **Connection Delete Probability**: The likelihood of a mutation that deletes an existing connection.
- **Node Add Probability**: The probability of a mutation that adds a new node to the network.
- **Node Delete Probability**: The probability of a mutation that deletes an existing node from the network.
- **Weight Mutate Rate**: The frequency at which the weights of connections are mutated.
- **Weight Mutate Power**: The magnitude of change applied to weights during mutation.
- **Max Stagnation**: The number of generations a species is allowed to exist without improvement before it's considered stagnant and removed.

These parameters are critical for the evolution and optimization of the neural networks that learn to play Flappy Bird. They are adjusted to ensure a balance between exploration of new neural structures and the exploitation of existing, well-performing structures.
