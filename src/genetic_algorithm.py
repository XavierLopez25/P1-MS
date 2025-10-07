import numpy as np
import random
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
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
                 convergence_threshold: int = 100,
                 coordinates: Optional[List] = None):
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
            coordinates: City coordinates for visualization
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
        self.coordinates = coordinates

        # Statistics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_tour_history = []  # Track best tour per generation
        self.best_tour = None
        self.best_fitness = float('inf')
        self.generation_count = 0
        self.convergence_count = 0

        # Operators
        self.operators = GeneticOperators()

        # Visualization setup
        self.real_time_viz = False
        self.viz_callback = None

    def run(self, verbose: bool = True, real_time_viz: bool = False,
            viz_callback: Optional[Callable] = None) -> Dict:
        """
        Run the genetic algorithm

        Args:
            verbose: Whether to print progress information
            real_time_viz: Enable real-time visualization
            viz_callback: Custom visualization callback function

        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()

        # Set up visualization
        self.real_time_viz = real_time_viz
        self.viz_callback = viz_callback

        if real_time_viz and self.coordinates is not None:
            plt.ion()  # Enable interactive mode
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
            self.fig.suptitle('Real-time GA Evolution for TSP')

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

            # Real-time visualization
            if self.real_time_viz and self.coordinates is not None and generation % 5 == 0:
                self._update_real_time_visualization(generation)

            # Custom visualization callback
            if self.viz_callback:
                self.viz_callback(self, generation, population, fitness_scores)

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

        # Close real-time visualization
        if self.real_time_viz and self.coordinates is not None:
            plt.ioff()
            plt.show()

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
            'best_tour_history': self.best_tour_history,
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

        # Track best tour for this generation
        best_idx = fitness_scores.index(current_best_fitness)
        self.best_tour_history.append(population[best_idx].copy())

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
            self.best_tour_history = []
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

    def _update_real_time_visualization(self, generation: int):
        """Update real-time visualization during GA execution"""
        if self.coordinates is None:
            return

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Plot convergence (left panel)
        self.ax1.plot(self.best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
        self.ax1.plot(self.avg_fitness_history, 'r--', label='Avg Fitness', alpha=0.7)
        self.ax1.set_title(f'Convergence - Generation {generation}')
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Fitness (Distance)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Plot current best tour (right panel)
        if self.best_tour is not None:
            self._plot_tour_on_axis(self.ax2, self.best_tour,
                                   f'Best Tour - Gen {generation}\nDistance: {self.best_fitness:.2f}')

        # Update display
        plt.pause(0.01)

    def _plot_tour_on_axis(self, ax, tour: List[int], title: str):
        """Plot tour on given axis"""
        # Extract coordinates
        x_coords = [self.coordinates[i][1] for i in range(len(self.coordinates))]
        y_coords = [self.coordinates[i][2] for i in range(len(self.coordinates))]

        # Plot cities
        ax.scatter(x_coords, y_coords, c='red', s=30, zorder=2)

        # Plot tour
        tour_x = [x_coords[city] for city in tour] + [x_coords[tour[0]]]
        tour_y = [y_coords[city] for city in tour] + [y_coords[tour[0]]]
        ax.plot(tour_x, tour_y, 'b-', linewidth=1.5, alpha=0.8, zorder=1)

        # Mark start city
        ax.scatter(x_coords[tour[0]], y_coords[tour[0]], c='green', s=60, marker='s', zorder=3)

        ax.set_title(title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)