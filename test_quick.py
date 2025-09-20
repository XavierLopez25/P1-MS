import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tsp_parser import TSPParser
from genetic_algorithm import GeneticAlgorithmTSP

def quick_test():
    """Quick test to verify the implementation works"""
    print("Quick test of GA implementation...")

    # Test with eil101 (smaller run)
    parser = TSPParser()
    problem_data = parser.parse_file('eil101.tsp')

    print(f"Loaded {problem_data['name']} with {problem_data['dimension']} cities")

    # Small GA for quick test
    ga = GeneticAlgorithmTSP(
        distance_matrix=problem_data['distance_matrix'],
        population_size=50,
        elite_size=5,
        mutation_rate=0.02,
        crossover_rate=0.8,
        max_generations=100,
        tournament_size=3,
        mutation_type='2opt',
        convergence_threshold=20
    )

    result = ga.run(verbose=True)

    print(f"\nTest completed successfully!")
    print(f"Best solution found: {result['best_fitness']:.2f}")
    print(f"Time taken: {result['execution_time']:.2f} seconds")

    return result

if __name__ == "__main__":
    quick_test()