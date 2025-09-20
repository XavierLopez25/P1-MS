import os
import sys
from typing import Dict
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tsp_parser import TSPParser
from custom_scenario import create_fibonacci_spiral_scenario
from genetic_algorithm import GeneticAlgorithmTSP
from utils import TSPUtils

def test_scenario(scenario_name: str, problem_data: Dict, num_runs: int = 10) -> Dict:
    """
    Test genetic algorithm on a specific scenario

    Args:
        scenario_name: Name of the scenario
        problem_data: Problem data dictionary
        num_runs: Number of independent runs

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"TESTING SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    print(f"Cities: {problem_data['dimension']}")
    print(f"Distance type: {problem_data['edge_weight_type']}")

    # Configure GA parameters based on problem size
    num_cities = problem_data['dimension']
    population_size = max(50, min(200, num_cities * 2))
    max_generations = max(500, min(2000, num_cities * 10))

    # Initialize GA
    ga = GeneticAlgorithmTSP(
        distance_matrix=problem_data['distance_matrix'],
        population_size=population_size,
        elite_size=max(5, population_size // 10),
        mutation_rate=0.02,
        crossover_rate=0.8,
        max_generations=max_generations,
        tournament_size=3,
        mutation_type='2opt',
        convergence_threshold=100
    )

    # Run multiple times for statistics
    results = ga.multiple_runs(num_runs=num_runs, verbose=False)

    # Add scenario info to results
    results['scenario_info'] = {
        'name': scenario_name,
        'num_cities': num_cities,
        'distance_type': problem_data['edge_weight_type']
    }

    # Print summary
    print(TSPUtils.create_results_summary(scenario_name, problem_data, results))

    return results

def run_all_experiments():
    """Run experiments on all three scenarios"""
    results_dict = {}

    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)

    # Scenario 1: eil101
    print("Loading eil101.tsp...")
    parser = TSPParser()
    eil101_data = parser.parse_file('../eil101.tsp')
    results_dict['eil101'] = test_scenario('EIL101', eil101_data, num_runs=10)

    # Save results
    TSPUtils.save_results(results_dict['eil101'], '../results/eil101_results.json')

    # Scenario 2: gr229
    print("\nLoading gr229.tsp...")
    parser = TSPParser()
    gr229_data = parser.parse_file('../gr229.tsp')
    results_dict['gr229'] = test_scenario('GR229', gr229_data, num_runs=10)

    # Save results
    TSPUtils.save_results(results_dict['gr229'], '../results/gr229_results.json')

    # Scenario 3: Custom Fibonacci Spiral
    print("\nGenerating Fibonacci Spiral scenario...")
    custom_data = create_fibonacci_spiral_scenario(100, '../fibonacci_spiral_100.tsp')
    results_dict['fibonacci_spiral'] = test_scenario('Fibonacci Spiral', custom_data, num_runs=10)

    # Save results
    TSPUtils.save_results(results_dict['fibonacci_spiral'], '../results/custom_results.json')

    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    comparison_table = TSPUtils.create_comparison_table(results_dict)
    print(comparison_table)

    # Save comparison table
    with open('../results/comparison_table.md', 'w') as f:
        f.write("# Comparison of GA Results Across Scenarios\n\n")
        f.write(comparison_table)

    # Generate convergence plots
    print("\nGenerating convergence plots...")
    for scenario_name, results in results_dict.items():
        plot_path = f'../results/{scenario_name}_convergence.png'
        TSPUtils.plot_convergence(results, f'GA Convergence - {scenario_name}', plot_path)

    # Generate tour visualizations for best solutions
    print("\nGenerating tour visualizations...")
    scenarios_data = {
        'eil101': eil101_data,
        'gr229': gr229_data,
        'fibonacci_spiral': custom_data
    }

    for scenario_name, results in results_dict.items():
        problem_data = scenarios_data[scenario_name]
        best_tour = results['best_tour']
        plot_path = f'../results/{scenario_name}_best_tour.png'
        TSPUtils.plot_tour(
            problem_data['coordinates'],
            best_tour,
            f'Best Tour - {scenario_name} (Distance: {results["best_fitness"]:.2f})',
            plot_path
        )

    print(f"\nAll results saved to '../results/' directory")
    return results_dict

def demo_single_run():
    """Demonstrate a single run with detailed output"""
    print("="*60)
    print("DEMO: Single Run on EIL101")
    print("="*60)

    # Load eil101
    parser = TSPParser()
    problem_data = parser.parse_file('../eil101.tsp')

    # Configure GA
    ga = GeneticAlgorithmTSP(
        distance_matrix=problem_data['distance_matrix'],
        population_size=100,
        elite_size=10,
        mutation_rate=0.02,
        crossover_rate=0.8,
        max_generations=500,
        tournament_size=3,
        mutation_type='2opt',
        convergence_threshold=50
    )

    # Run with verbose output
    result = ga.run(verbose=True)

    # Display results
    print(f"\nFinal Results:")
    print(f"Best distance: {result['best_fitness']:.2f}")
    print(f"Execution time: {result['execution_time']:.2f} seconds")
    print(f"Generations: {result['generations']}")

    # Plot convergence
    TSPUtils.plot_convergence(result, "EIL101 Single Run Convergence")

    # Plot best tour
    TSPUtils.plot_tour(
        problem_data['coordinates'],
        result['best_tour'],
        f"EIL101 Best Tour (Distance: {result['best_fitness']:.2f})"
    )

def main():
    """Main function"""
    print("Genetic Algorithm for TSP - Project Implementation")
    print("="*60)

    choice = input("\nSelect option:\n1. Run all experiments\n2. Demo single run\n3. Exit\nChoice: ")

    if choice == '1':
        print("\nRunning comprehensive experiments on all scenarios...")
        print("This may take several minutes...")
        run_all_experiments()

    elif choice == '2':
        demo_single_run()

    elif choice == '3':
        print("Goodbye!")

    else:
        print("Invalid choice!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()