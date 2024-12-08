# train.py

import numpy as np
import pandas as pd
import random
import os
from copy import deepcopy
from game import Game
from custom_model import CUSTOM_AI_MODEL

def cross(a1, a2):
    """
    Crossover between two parent genotypes to produce a child genotype.
    """
    new_genotype = np.copy(a1.genotype)
    for i in range(len(a1.genotype)):
        if random.random() < 0.5:
            new_genotype[i] = a2.genotype[i]
    return new_genotype

def mutate(genotype, mutation_rate=0.1, mutation_strength=0.5):
    """
    Mutates the genotype by adding Gaussian noise.
    """
    for i in range(len(genotype)):
        if random.random() < mutation_rate:
            genotype[i] += random.gauss(0, mutation_strength)
            # Clamp the values to a reasonable range
            genotype[i] = max(-10, min(10, genotype[i]))
    return genotype

def compute_fitness(model, num_trials=5):
    """
    Computes the fitness of a model by averaging the number of pieces dropped over multiple trials.
    """
    total_dropped = 0
    for _ in range(num_trials):
        game = Game(mode='student', agent=model)
        dropped, _ = game.run_no_visual()
        total_dropped += dropped
    return total_dropped / num_trials

def run_X_epochs(num_epochs=10, num_trials=5, pop_size=100, num_elite=5, mutation_rate=0.1, mutation_strength=0.5, weights_path='trained_model.npy'):
    """
    Runs the genetic algorithm for a specified number of epochs.
    """
    # Initialize population with random models
    population = [CUSTOM_AI_MODEL() for _ in range(pop_size)]
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Evaluate fitness
        for i, model in enumerate(population):
            model.fit_score = compute_fitness(model, num_trials=num_trials)
            print(f"  Agent {i + 1}/{pop_size}: Fitness = {model.fit_score}")
        
        # Sort population by descending fitness
        population.sort(key=lambda x: x.fit_score, reverse=True)
        
        # Elitism: carry over the top-performing agents
        elites = population[:num_elite]
        print(f"  Top {num_elite} elites: Fitness = {[elite.fit_score for elite in elites]}")
        
        # Selection: select parents based on fitness proportionate selection (roulette wheel)
        fitness_sum = sum([agent.fit_score for agent in population])
        if fitness_sum == 0:
            probabilities = [1/pop_size] * pop_size
        else:
            probabilities = [agent.fit_score / fitness_sum for agent in population]
        
        parents = np.random.choice(population, size=pop_size - num_elite, p=probabilities)
        
        # Create next generation
        next_gen = elites.copy()
        while len(next_gen) < pop_size:
            parent1, parent2 = random.sample(list(parents), 2)
            child_genotype = cross(parent1, parent2)
            child_genotype = mutate(child_genotype, mutation_rate, mutation_strength)
            child = CUSTOM_AI_MODEL()
            child.genotype = child_genotype
            next_gen.append(child)
        
        population = next_gen
        
        # Optionally, save the best model each epoch
        best_model = population[0]
        best_model.save_weights(weights_path)
        print(f"  Saved best model of epoch {epoch + 1} with Fitness = {best_model.fit_score}")
    
    print("\nTraining completed.")
    print(f"Best model saved to {weights_path}")

if __name__ == "__main__":
    run_X_epochs(
        num_epochs=15,
        num_trials=5,
        pop_size=50,
        num_elite=5,
        mutation_rate=0.1,
        mutation_strength=0.5,
        weights_path='trained_model.npy'
    )
