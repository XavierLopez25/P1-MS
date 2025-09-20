import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os

class TSPUtils:
    """Utility functions for TSP analysis and visualization"""

    @staticmethod
    def save_results(results: Dict, filename: str):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = TSPUtils._convert_for_json(results)

        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)

    @staticmethod
    def _convert_for_json(obj):
        """Convert numpy arrays and other objects to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: TSPUtils._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [TSPUtils._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            return obj

    @staticmethod
    def load_results(filename: str) -> Dict:
        """Load results from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)

    @staticmethod
    def plot_convergence(results: Dict, title: str = "GA Convergence", save_path: str = None):
        """Plot convergence graph"""
        plt.figure(figsize=(12, 6))

        if 'all_results' in results:
            # Multiple runs - plot all and average
            plt.subplot(1, 2, 1)
            for i, result in enumerate(results['all_results']):
                plt.plot(result['best_fitness_history'], alpha=0.3, color='blue')
            plt.title('All Runs - Best Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            # Calculate average convergence
            max_len = max(len(r['best_fitness_history']) for r in results['all_results'])
            avg_convergence = []
            for gen in range(max_len):
                gen_values = []
                for result in results['all_results']:
                    if gen < len(result['best_fitness_history']):
                        gen_values.append(result['best_fitness_history'][gen])
                    else:
                        gen_values.append(result['best_fitness_history'][-1])
                avg_convergence.append(np.mean(gen_values))

            plt.plot(avg_convergence, color='red', linewidth=2)
            plt.title('Average Convergence')
            plt.xlabel('Generation')
            plt.ylabel('Average Best Fitness')
            plt.grid(True)
        else:
            # Single run
            plt.plot(results['best_fitness_history'], label='Best Fitness', color='blue')
            plt.plot(results['avg_fitness_history'], label='Average Fitness', color='orange')
            plt.title(title)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_tour(coordinates: List, tour: List[int], title: str = "Best Tour", save_path: str = None):
        """Plot the tour on a map"""
        plt.figure(figsize=(10, 8))

        # Extract coordinates
        x_coords = [coordinates[i][1] for i in range(len(coordinates))]
        y_coords = [coordinates[i][2] for i in range(len(coordinates))]

        # Plot cities
        plt.scatter(x_coords, y_coords, c='red', s=50, zorder=2)

        # Plot tour
        tour_x = [x_coords[city] for city in tour] + [x_coords[tour[0]]]
        tour_y = [y_coords[city] for city in tour] + [y_coords[tour[0]]]
        plt.plot(tour_x, tour_y, 'b-', linewidth=1, alpha=0.7, zorder=1)

        # Mark start city
        plt.scatter(x_coords[tour[0]], y_coords[tour[0]], c='green', s=100, marker='s', zorder=3)

        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def create_comparison_table(results_dict: Dict[str, Dict]) -> str:
        """Create comparison table for different scenarios"""
        table = "| Escenario | Mejor Solución | Tiempo Promedio | Generaciones | Desviación Estándar |\n"
        table += "|-----------|----------------|-----------------|--------------|--------------------|\n"

        for scenario_name, results in results_dict.items():
            if 'all_results' in results:
                # Multiple runs
                best_fitness = results['best_fitness']
                avg_time = results['mean_execution_time']
                avg_generations = results['mean_generations']
                std_fitness = results['std_fitness']
            else:
                # Single run
                best_fitness = results['best_fitness']
                avg_time = results['execution_time']
                avg_generations = results['generations']
                std_fitness = 0

            table += f"| {scenario_name} | {best_fitness:.2f} | {avg_time:.2f}s | {avg_generations:.0f} | {std_fitness:.2f} |\n"

        return table

    @staticmethod
    def validate_tour(tour: List[int], num_cities: int) -> bool:
        """Validate that a tour visits all cities exactly once"""
        if len(tour) != num_cities:
            return False

        if len(set(tour)) != num_cities:
            return False

        if min(tour) != 0 or max(tour) != num_cities - 1:
            return False

        return True

    @staticmethod
    def calculate_tour_length(tour: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate the total length of a tour"""
        total_distance = 0.0
        num_cities = len(tour)

        for i in range(num_cities):
            current_city = tour[i]
            next_city = tour[(i + 1) % num_cities]
            total_distance += distance_matrix[current_city][next_city]

        return total_distance

    @staticmethod
    def create_results_summary(scenario_name: str, problem_data: Dict, results: Dict) -> str:
        """Create a formatted summary of results"""
        summary = f"\n=== RESULTADOS PARA {scenario_name.upper()} ===\n"
        summary += f"Número de ciudades: {problem_data['dimension']}\n"
        summary += f"Tipo de problema: {problem_data['type']}\n"
        summary += f"Tipo de distancia: {problem_data['edge_weight_type']}\n\n"

        if 'all_results' in results:
            summary += f"Número de ejecuciones: {results['num_runs']}\n"
            summary += f"Mejor solución encontrada: {results['best_fitness']:.2f}\n"
            summary += f"Peor solución encontrada: {results['worst_fitness']:.2f}\n"
            summary += f"Solución promedio: {results['mean_fitness']:.2f}\n"
            summary += f"Desviación estándar: {results['std_fitness']:.2f}\n"
            summary += f"Tiempo promedio de ejecución: {results['mean_execution_time']:.2f} segundos\n"
            summary += f"Generaciones promedio: {results['mean_generations']:.0f}\n"
            summary += f"Tasa de convergencia: {results['success_rate']*100:.1f}%\n"
        else:
            summary += f"Mejor solución encontrada: {results['best_fitness']:.2f}\n"
            summary += f"Tiempo de ejecución: {results['execution_time']:.2f} segundos\n"
            summary += f"Generaciones: {results['generations']}\n"

        summary += "\n" + "="*50 + "\n"

        return summary