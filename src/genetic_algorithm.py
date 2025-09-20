import numpy as np
import random
import time
from typing import List, Dict, Tuple, Optional
from operators import GeneticOperators

class GeneticAlgorithmTSP:
    """Genetic Algorithm implementation for the Traveling Salesman Problem"""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 population_size: int = 100,
                 elite_size: int = 10,
                 mutation_rate: float = 0.02,
                 crossover_rate: float = 0.8,
                 max_generations: int = 1000,
                 tournament_size: int = 3,
                 mutation_type: str = '2opt',
                 convergence_threshold: int = 100):
        """
        Initialize the Genetic Algorithm

        Args:
            distance_matrix: Matrix of distances between cities
            population_size: Size of the population
            elite_size: Number of elite individuals to preserve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            max_generations: Maximum number of generations
            tournament_size: Size of tournament for selection
            mutation_type: Type of mutation ('2opt', 'swap', 'inversion')
            convergence_threshold: Generations without improvement to stop
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.mutation_type = mutation_type
        self.convergence_threshold = convergence_threshold

        # Statistics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_tour = None
        self.best_fitness = float('inf')
        self.generation_count = 0
        self.convergence_count = 0

        # Operators
        self.operators = GeneticOperators()

    def run(self, verbose: bool = True) -> Dict:
        """
        Run the genetic algorithm

        Args:
            verbose: Whether to print progress information

        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()

        # Initialize population
        population = self.operators.create_initial_population(
            self.num_cities, self.population_size
        )

        if verbose:
            print(f"Starting GA with {self.num_cities} cities")
            print(f"Population size: {self.population_size}")
            print(f"Elite size: {self.elite_size}")
            print(f"Mutation rate: {self.mutation_rate}")
            print(f"Max generations: {self.max_generations}")

        # Main evolution loop
        for generation in range(self.max_generations):
            self.generation_count = generation

            # Evaluate fitness
            fitness_scores = self._evaluate_population(population)

            # Track statistics
            self._update_statistics(population, fitness_scores)

            # Check for convergence
            if self._check_convergence():
                if verbose:
                    print(f"Converged at generation {generation}")
                break

            # Create next generation
            population = self._create_next_generation(population, fitness_scores)

            # Print progress
            if verbose and generation % 50 == 0:
                print(f"Generation {generation}: Best = {self.best_fitness:.2f}, "
                      f"Avg = {self.avg_fitness_history[-1]:.2f}")

        end_time = time.time()
        execution_time = end_time - start_time

        if verbose:
            print(f"Algorithm completed in {execution_time:.2f} seconds")
            print(f"Best tour distance: {self.best_fitness:.2f}")

        return {
            'best_tour': self.best_tour,
            'best_fitness': self.best_fitness,
            'execution_time': execution_time,
            'generations': self.generation_count + 1,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'convergence_generation': self.generation_count if self.convergence_count >= self.convergence_threshold else None
        }

    def _evaluate_population(self, population: List[List[int]]) -> List[float]:
        """Evaluate fitness of all individuals in population"""
        fitness_scores = []
        for individual in population:
            fitness = self.operators.calculate_fitness(individual, self.distance_matrix)
            fitness_scores.append(fitness)
        return fitness_scores

    def _update_statistics(self, population: List[List[int]], fitness_scores: List[float]):
        """Update statistics and track best solution"""
        current_best_fitness = min(fitness_scores)
        current_avg_fitness = np.mean(fitness_scores)

        self.best_fitness_history.append(current_best_fitness)
        self.avg_fitness_history.append(current_avg_fitness)

        # Update best solution if improved
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            best_idx = fitness_scores.index(current_best_fitness)
            self.best_tour = population[best_idx].copy()
            self.convergence_count = 0
        else:
            self.convergence_count += 1

    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""
        return self.convergence_count >= self.convergence_threshold

    def _create_next_generation(self, population: List[List[int]],
                               fitness_scores: List[float]) -> List[List[int]]:
        """Create the next generation using genetic operators"""
        next_generation = []

        # Elitism: preserve best individuals
        elite_individuals = self.operators.elitism_selection(
            population, fitness_scores, self.elite_size
        )
        next_generation.extend(elite_individuals)

        # Generate remaining individuals through crossover and mutation
        while len(next_generation) < self.population_size:
            # Selection
            parent1 = self.operators.tournament_selection(
                population, fitness_scores, self.tournament_size
            )
            parent2 = self.operators.tournament_selection(
                population, fitness_scores, self.tournament_size
            )

            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.operators.order_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()

            # Mutation
            offspring1 = self._apply_mutation(offspring1)
            offspring2 = self._apply_mutation(offspring2)

            # Add offspring to next generation
            next_generation.append(offspring1)
            if len(next_generation) < self.population_size:
                next_generation.append(offspring2)

        return next_generation[:self.population_size]

    def _apply_mutation(self, individual: List[int]) -> List[int]:
        """Apply mutation based on selected mutation type"""
        if self.mutation_type == '2opt':
            return self.operators.two_opt_mutation(individual, self.mutation_rate)
        elif self.mutation_type == 'swap':
            return self.operators.swap_mutation(individual, self.mutation_rate)
        elif self.mutation_type == 'inversion':
            return self.operators.inversion_mutation(individual, self.mutation_rate)
        else:
            raise ValueError(f"Unknown mutation type: {self.mutation_type}")

    def get_tour_distance(self, tour: List[int]) -> float:
        """Calculate total distance of a tour"""
        return self.operators.calculate_fitness(tour, self.distance_matrix)

    def multiple_runs(self, num_runs: int = 10, verbose: bool = False) -> Dict:
        """
        Run the algorithm multiple times and return statistics

        Args:
            num_runs: Number of independent runs
            verbose: Whether to print detailed progress

        Returns:
            Dictionary with aggregated statistics
        """
        results = []
        best_overall_fitness = float('inf')
        best_overall_tour = None

        print(f"Running {num_runs} independent executions...")

        for run in range(num_runs):
            if verbose:
                print(f"\n--- Run {run + 1}/{num_runs} ---")

            # Reset for new run
            self.best_fitness = float('inf')
            self.best_tour = None
            self.best_fitness_history = []
            self.avg_fitness_history = []
            self.convergence_count = 0

            # Run algorithm
            result = self.run(verbose=verbose)
            results.append(result)

            # Track overall best
            if result['best_fitness'] < best_overall_fitness:
                best_overall_fitness = result['best_fitness']
                best_overall_tour = result['best_tour']

            if not verbose:
                print(f"Run {run + 1}: {result['best_fitness']:.2f}")

        # Calculate statistics
        fitnesses = [r['best_fitness'] for r in results]
        execution_times = [r['execution_time'] for r in results]
        generations = [r['generations'] for r in results]

        return {
            'num_runs': num_runs,
            'best_fitness': min(fitnesses),
            'worst_fitness': max(fitnesses),
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_tour': best_overall_tour,
            'mean_execution_time': np.mean(execution_times),
            'mean_generations': np.mean(generations),
            'all_results': results,
            'success_rate': sum(1 for r in results if r['convergence_generation'] is not None) / num_runs
        }