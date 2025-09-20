import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tsp_parser import TSPParser
from custom_scenario import create_fibonacci_spiral_scenario
from genetic_algorithm import GeneticAlgorithmTSP
from utils import TSPUtils

def run_quick_experiments():
    """Run quick experiments on all scenarios (3 runs each for demo)"""
    results_dict = {}

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    print("Running quick experiments on all 3 scenarios...")
    print("="*60)

    # Scenario 1: eil101
    print("\n1. Testing EIL101 scenario...")
    parser = TSPParser()
    eil101_data = parser.parse_file('eil101.tsp')

    ga_eil101 = GeneticAlgorithmTSP(
        distance_matrix=eil101_data['distance_matrix'],
        population_size=100,
        elite_size=10,
        mutation_rate=0.02,
        crossover_rate=0.8,
        max_generations=200,
        tournament_size=3,
        mutation_type='2opt',
        convergence_threshold=50
    )

    results_dict['eil101'] = ga_eil101.multiple_runs(num_runs=3, verbose=False)
    TSPUtils.save_results(results_dict['eil101'], 'results/eil101_results.json')
    print(f"EIL101 - Best: {results_dict['eil101']['best_fitness']:.2f}")

    # Scenario 2: gr229
    print("\n2. Testing GR229 scenario...")
    parser = TSPParser()
    gr229_data = parser.parse_file('gr229.tsp')

    ga_gr229 = GeneticAlgorithmTSP(
        distance_matrix=gr229_data['distance_matrix'],
        population_size=150,
        elite_size=15,
        mutation_rate=0.02,
        crossover_rate=0.8,
        max_generations=300,
        tournament_size=3,
        mutation_type='2opt',
        convergence_threshold=50
    )

    results_dict['gr229'] = ga_gr229.multiple_runs(num_runs=3, verbose=False)
    TSPUtils.save_results(results_dict['gr229'], 'results/gr229_results.json')
    print(f"GR229 - Best: {results_dict['gr229']['best_fitness']:.2f}")

    # Scenario 3: Custom Fibonacci Spiral
    print("\n3. Testing Fibonacci Spiral scenario...")
    custom_data = create_fibonacci_spiral_scenario(100, 'fibonacci_spiral_100.tsp')

    ga_custom = GeneticAlgorithmTSP(
        distance_matrix=custom_data['distance_matrix'],
        population_size=120,
        elite_size=12,
        mutation_rate=0.02,
        crossover_rate=0.8,
        max_generations=250,
        tournament_size=3,
        mutation_type='2opt',
        convergence_threshold=50
    )

    results_dict['fibonacci_spiral'] = ga_custom.multiple_runs(num_runs=3, verbose=False)
    TSPUtils.save_results(results_dict['fibonacci_spiral'], 'results/custom_results.json')
    print(f"Fibonacci Spiral - Best: {results_dict['fibonacci_spiral']['best_fitness']:.2f}")

    # Create comparison table
    print("\n" + "="*60)
    print("QUICK EXPERIMENT RESULTS")
    print("="*60)
    comparison_table = TSPUtils.create_comparison_table(results_dict)
    print(comparison_table)

    # Save comparison table
    with open('results/quick_comparison_table.md', 'w') as f:
        f.write("# Quick Experiment Results - GA for TSP\n\n")
        f.write("## Comparison Table\n\n")
        f.write(comparison_table)
        f.write("\n## Scenario Details\n\n")
        f.write("- **EIL101**: 101 cities, Euclidean distances\n")
        f.write("- **GR229**: 229 cities, Geographical distances\n")
        f.write("- **Fibonacci Spiral**: 100 cities arranged in Fibonacci spiral pattern\n")
        f.write("\n## Algorithm Parameters\n\n")
        f.write("- Selection: Tournament selection (size=3)\n")
        f.write("- Crossover: Order Crossover (OX)\n")
        f.write("- Mutation: 2-opt mutation\n")
        f.write("- Elitism: 10% of population\n")

    print(f"\nResults saved to 'results/' directory")
    print(f"Quick experiments completed successfully!")

    return results_dict

if __name__ == "__main__":
    run_quick_experiments()