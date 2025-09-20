import numpy as np
import random
from typing import List, Tuple

class GeneticOperators:
    """Collection of genetic operators for TSP"""

    @staticmethod
    def tournament_selection(population: List[List[int]],
                           fitness_scores: List[float],
                           tournament_size: int = 3) -> List[int]:
        """
        Tournament selection operator

        Args:
            population: List of tours (each tour is a list of city indices)
            fitness_scores: List of fitness scores (lower is better for TSP)
            tournament_size: Number of individuals in each tournament

        Returns:
            Selected individual (tour)
        """
        # Select random individuals for tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)

        # Find the best individual in tournament (lowest fitness for TSP)
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])

        return population[best_idx].copy()

    @staticmethod
    def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX) operator for TSP

        Args:
            parent1: First parent tour
            parent2: Second parent tour

        Returns:
            Tuple of two offspring tours
        """
        size = len(parent1)

        # Choose two random crossover points
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)

        # Create offspring
        offspring1 = [-1] * size
        offspring2 = [-1] * size

        # Copy the segment between crossover points
        offspring1[start:end] = parent1[start:end]
        offspring2[start:end] = parent2[start:end]

        # Fill remaining positions with cities from other parent in order
        GeneticOperators._fill_offspring(offspring1, parent2, start, end)
        GeneticOperators._fill_offspring(offspring2, parent1, start, end)

        return offspring1, offspring2

    @staticmethod
    def _fill_offspring(offspring: List[int], parent: List[int], start: int, end: int):
        """Helper function for order crossover"""
        size = len(offspring)
        parent_idx = end % size
        offspring_idx = end % size

        while -1 in offspring:
            if parent[parent_idx] not in offspring:
                offspring[offspring_idx] = parent[parent_idx]
                offspring_idx = (offspring_idx + 1) % size
            parent_idx = (parent_idx + 1) % size

    @staticmethod
    def two_opt_mutation(tour: List[int], mutation_rate: float = 0.1) -> List[int]:
        """
        2-opt mutation operator

        Args:
            tour: Tour to mutate
            mutation_rate: Probability of mutation

        Returns:
            Mutated tour
        """
        if random.random() > mutation_rate:
            return tour.copy()

        mutated_tour = tour.copy()
        size = len(tour)

        # Choose two random positions
        i = random.randint(0, size - 2)
        j = random.randint(i + 1, size - 1)

        # Reverse the segment between i and j
        mutated_tour[i:j+1] = reversed(mutated_tour[i:j+1])

        return mutated_tour

    @staticmethod
    def swap_mutation(tour: List[int], mutation_rate: float = 0.1) -> List[int]:
        """
        Swap mutation operator

        Args:
            tour: Tour to mutate
            mutation_rate: Probability of mutation

        Returns:
            Mutated tour
        """
        if random.random() > mutation_rate:
            return tour.copy()

        mutated_tour = tour.copy()
        size = len(tour)

        # Choose two random positions
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        # Swap the cities
        mutated_tour[i], mutated_tour[j] = mutated_tour[j], mutated_tour[i]

        return mutated_tour

    @staticmethod
    def inversion_mutation(tour: List[int], mutation_rate: float = 0.1) -> List[int]:
        """
        Inversion mutation operator

        Args:
            tour: Tour to mutate
            mutation_rate: Probability of mutation

        Returns:
            Mutated tour
        """
        if random.random() > mutation_rate:
            return tour.copy()

        mutated_tour = tour.copy()
        size = len(tour)

        # Choose a random segment to invert
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size)

        # Reverse the segment
        mutated_tour[start:end] = reversed(mutated_tour[start:end])

        return mutated_tour

    @staticmethod
    def create_initial_population(num_cities: int, population_size: int) -> List[List[int]]:
        """
        Create initial population with random tours

        Args:
            num_cities: Number of cities in the problem
            population_size: Size of the population

        Returns:
            List of random tours
        """
        population = []
        base_tour = list(range(num_cities))

        for _ in range(population_size):
            tour = base_tour.copy()
            random.shuffle(tour)
            population.append(tour)

        return population

    @staticmethod
    def calculate_fitness(tour: List[int], distance_matrix: np.ndarray) -> float:
        """
        Calculate fitness (total distance) of a tour

        Args:
            tour: Tour to evaluate
            distance_matrix: Distance matrix between cities

        Returns:
            Total distance of the tour
        """
        total_distance = 0.0
        num_cities = len(tour)

        for i in range(num_cities):
            current_city = tour[i]
            next_city = tour[(i + 1) % num_cities]  # Return to start city
            total_distance += distance_matrix[current_city][next_city]

        return total_distance

    @staticmethod
    def elitism_selection(population: List[List[int]],
                         fitness_scores: List[float],
                         elite_size: int) -> List[List[int]]:
        """
        Select the best individuals for elitism

        Args:
            population: Current population
            fitness_scores: Fitness scores (lower is better)
            elite_size: Number of elite individuals to select

        Returns:
            List of elite individuals
        """
        # Sort population by fitness (ascending for minimization)
        elite_indices = sorted(range(len(fitness_scores)),
                             key=lambda i: fitness_scores[i])[:elite_size]

        return [population[i].copy() for i in elite_indices]