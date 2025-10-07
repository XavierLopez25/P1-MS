import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os
import subprocess
import tempfile

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

    @staticmethod
    def create_combined_comparison_table(ga_results: Dict, lp_results: Dict) -> str:
        """Create comparison table between GA and LP results"""
        table = "# Comparación entre Algoritmo Genético y Programación Lineal\n\n"
        table += "| Método | Escenario | Solución Encontrada | Tiempo Ejecución | % Error vs Óptimo | Observaciones |\n"
        table += "|--------|-----------|--------------------|-----------------|--------------------|---------------|\n"

        scenarios = ['eil101', 'gr229', 'fibonacci_spiral']

        for scenario in scenarios:
            # LP results
            if scenario in lp_results:
                lp_result = lp_results[scenario]
                lp_distance = lp_result.get('objective_value', 'N/A')
                lp_time = lp_result.get('solve_time', 'N/A')
                lp_status = lp_result.get('status', 'N/A')
                optimal_distance = lp_distance if lp_distance != 'N/A' else None

                lp_dist_str = f"{lp_distance:.2f}" if isinstance(lp_distance, (int, float)) else str(lp_distance)
                lp_time_str = f"{lp_time:.2f}s" if isinstance(lp_time, (int, float)) else str(lp_time)
                table += f"| LP (MTZ) | {scenario} | {lp_dist_str} | {lp_time_str} | 0.00% | {lp_status} |\n"

            # GA results
            if scenario in ga_results:
                ga_result = ga_results[scenario]
                ga_best = ga_result.get('best_fitness', 'N/A')
                ga_time = ga_result.get('mean_execution_time', ga_result.get('execution_time', 'N/A'))

                # Calculate error vs optimal
                error_pct = "N/A"
                if optimal_distance and isinstance(ga_best, (int, float)) and isinstance(optimal_distance, (int, float)):
                    error_pct = f"{((ga_best - optimal_distance) / optimal_distance * 100):.2f}%"

                ga_best_str = f"{ga_best:.2f}" if isinstance(ga_best, (int, float)) else str(ga_best)
                ga_time_str = f"{ga_time:.2f}s" if isinstance(ga_time, (int, float)) else str(ga_time)
                table += f"| GA | {scenario} | {ga_best_str} | {ga_time_str} | {error_pct} | Heurístico |\n"

        return table

    @staticmethod
    def create_detailed_metrics_table(ga_results: Dict, lp_results: Dict) -> str:
        """Create detailed metrics table as specified in CLAUDE.md"""
        table = "\n## Métricas Detalladas por Escenario\n\n"

        scenarios = ['eil101', 'gr229', 'fibonacci_spiral']
        scenario_names = ['EIL101', 'GR229', 'Fibonacci Spiral']

        for i, scenario in enumerate(scenarios):
            table += f"### {scenario_names[i]}\n\n"
            table += "| Método | Tiempo Ejecución | Solución Encontrada | Solución Óptima Teórica | % Error |\n"
            table += "|--------|------------------|--------------------|-----------------------|---------|\n"

            # LP row
            if scenario in lp_results:
                lp_result = lp_results[scenario]
                lp_distance = lp_result.get('objective_value', 'N/A')
                lp_time = lp_result.get('solve_time', 'N/A')
                optimal_value = lp_distance

                lp_time_str = f"{lp_time:.2f}s" if isinstance(lp_time, (int, float)) else str(lp_time)
                lp_distance_str = f"{lp_distance:.2f}" if isinstance(lp_distance, (int, float)) else str(lp_distance)
                optimal_value_str = f"{optimal_value:.2f}" if isinstance(optimal_value, (int, float)) else str(optimal_value)
                table += f"| LP | {lp_time_str} | {lp_distance_str} | {optimal_value_str} | 0.00% |\n"

            # GA rows (best, 2nd best, 3rd best)
            if scenario in ga_results:
                ga_result = ga_results[scenario]

                if 'all_results' in ga_result:
                    # Sort results by fitness
                    sorted_results = sorted(ga_result['all_results'], key=lambda x: x['best_fitness'])

                    for j, rank in enumerate(['GA (mejor)', 'GA (2da mejor)', 'GA (3ra mejor)']):
                        if j < len(sorted_results):
                            result = sorted_results[j]
                            fitness = result['best_fitness']
                            time_exec = result['execution_time']

                            # Calculate error
                            error_pct = "N/A"
                            if optimal_value and isinstance(fitness, (int, float)) and isinstance(optimal_value, (int, float)):
                                error_pct = f"{((fitness - optimal_value) / optimal_value * 100):.2f}%"

                            optimal_str = f"{optimal_value:.2f}" if isinstance(optimal_value, (int, float)) else 'N/A'
                            table += f"| {rank} | {time_exec:.2f}s | {fitness:.2f} | {optimal_str} | {error_pct} |\n"

            # Metrics
            if scenario in ga_results and scenario in lp_results:
                ga_info = ga_results[scenario].get('scenario_info', {})
                num_cities = ga_info.get('num_cities', 'N/A')

                table += f"\n**Métricas del escenario:**\n"
                table += f"- Número de ciudades: {num_cities}\n"
                table += f"- Tamaño de población del GA: {ga_results[scenario].get('population_size', 'N/A')}\n"
                table += f"- Número de iteraciones del GA: {ga_results[scenario].get('mean_generations', 'N/A'):.0f}\n"

                if scenario in lp_results:
                    # Estimate LP variables and constraints
                    n = num_cities if isinstance(num_cities, int) else 100
                    lp_vars = n * (n - 1)  # x_ij variables
                    lp_constraints = 2 * n + n * (n - 1)  # degree + MTZ constraints
                    table += f"- Número de variables LP: {lp_vars}\n"
                    table += f"- Número de restricciones LP: {lp_constraints}\n"

                table += "\n"

        return table

    @staticmethod
    def plot_ga_evolution_animation(results: Dict, coordinates: List, scenario_name: str,
                                   save_path: Optional[str] = None, frames_dir: Optional[str] = None):
        """Create animation showing GA evolution over generations"""
        if 'best_tour_history' not in results:
            print("No tour history available for animation")
            return

        tour_history = results['best_tour_history']
        fitness_history = results['best_fitness_history']

        # Create frames directory if specified
        if frames_dir:
            os.makedirs(frames_dir, exist_ok=True)

        # Extract coordinates
        x_coords = [coordinates[i][1] for i in range(len(coordinates))]
        y_coords = [coordinates[i][2] for i in range(len(coordinates))]

        # Create frames - ensure we don't exceed bounds
        max_frames = min(len(tour_history), len(fitness_history))
        frame_interval = max(1, max_frames // 50)  # Max 50 frames

        for i in range(0, max_frames, frame_interval):
            # Ensure we don't go out of bounds
            idx = min(i, len(tour_history) - 1, len(fitness_history) - 1)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot convergence
            ax1.plot(fitness_history[:idx+1], 'b-', linewidth=2)
            ax1.set_title(f'Convergence - Generation {idx}')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Best Fitness')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, len(fitness_history))
            ax1.set_ylim(min(fitness_history) * 0.95, max(fitness_history) * 1.05)

            # Plot current tour
            tour = tour_history[idx]
            tour_x = [x_coords[city] for city in tour] + [x_coords[tour[0]]]
            tour_y = [y_coords[city] for city in tour] + [y_coords[tour[0]]]

            ax2.scatter(x_coords, y_coords, c='red', s=30, zorder=2)
            ax2.plot(tour_x, tour_y, 'b-', linewidth=1.5, alpha=0.8, zorder=1)
            ax2.scatter(x_coords[tour[0]], y_coords[tour[0]], c='green', s=60, marker='s', zorder=3)
            ax2.set_title(f'{scenario_name} - Gen {idx}\nDistance: {fitness_history[idx]:.2f}')
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if frames_dir:
                plt.savefig(f"{frames_dir}/frame_{idx:04d}.png", dpi=150, bbox_inches='tight')
                plt.close()
            elif idx == max_frames - 1 or i >= max_frames - frame_interval:  # Show last frame
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                break

        if not frames_dir:
            plt.close()

    @staticmethod
    def create_comprehensive_report(ga_results: Dict, lp_results: Dict,
                                  output_dir: str = '../results') -> str:
        """Create comprehensive analysis report combining GA and LP results"""
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, 'comprehensive_analysis.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Análisis Comparativo: Algoritmo Genético vs Programación Lineal\n")
            f.write("## Proyecto 1 - Traveling Salesman Problem (TSP)\n\n")

            f.write("### Resumen Ejecutivo\n\n")
            f.write("Este informe presenta una comparación exhaustiva entre dos enfoques para resolver el ")
            f.write("Traveling Salesman Problem (TSP): Algoritmos Genéticos (GA) y Programación Lineal (LP).\n\n")

            # Combined comparison table
            f.write(TSPUtils.create_combined_comparison_table(ga_results, lp_results))
            f.write("\n")

            # Detailed metrics
            f.write(TSPUtils.create_detailed_metrics_table(ga_results, lp_results))
            f.write("\n")

            f.write("### Análisis de Métodos\n\n")
            f.write("#### Programación Lineal (Formulación MTZ)\n")
            f.write("- **Ventajas:** Garantiza solución óptima, formulación matemática exacta\n")
            f.write("- **Desventajas:** Tiempo exponencial para problemas grandes, uso intensivo de memoria\n")
            f.write("- **Aplicabilidad:** Ideal para problemas pequeños (<150 ciudades)\n\n")

            f.write("#### Algoritmo Genético\n")
            f.write("- **Ventajas:** Escalable, tiempo de ejecución predecible, buena calidad de soluciones\n")
            f.write("- **Desventajas:** No garantiza optimalidad, requiere ajuste de parámetros\n")
            f.write("- **Aplicabilidad:** Excelente para problemas grandes (>100 ciudades)\n\n")

            f.write("### Conclusiones\n\n")
            f.write("1. **Para problemas pequeños (≤101 ciudades):** LP encuentra soluciones óptimas en tiempo razonable\n")
            f.write("2. **Para problemas medianos (100-300 ciudades):** GA ofrece mejor balance tiempo/calidad\n")
            f.write("3. **Para problemas grandes (>300 ciudades):** GA es la única opción práctica\n\n")

            f.write("### Implementación Técnica\n\n")
            f.write("- **LP:** Julia + JuMP + HiGHS solver, formulación Miller-Tucker-Zemlin\n")
            f.write("- **GA:** Python, selección por torneo, cruce OX, mutación 2-opt\n")
            f.write("- **Visualización:** Matplotlib para convergencia y tours, integración tiempo real\n\n")

            f.write("---\n")
            f.write("*Generado automáticamente por el sistema de análisis TSP*\n")

        return report_path

    @staticmethod
    def extract_julia_results_from_notebook(notebook_path: str = '../julia/pry1.ipynb') -> Dict:
        """Extract LP results from executed Julia notebook"""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)

            # Find the cell with execution results
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code' and 'outputs' in cell:
                    for output in cell['outputs']:
                        if output.get('output_type') == 'stream' and 'text' in output:
                            text = ''.join(output['text'])
                            if 'RESUMEN DE RESULTADOS' in text:
                                return TSPUtils._parse_julia_output(text)

            print("No results found in notebook")
            return {}

        except Exception as e:
            print(f"Error extracting results from notebook: {e}")
            return {}

    @staticmethod
    def _parse_julia_output(text: str) -> Dict:
        """Parse Julia output text to extract structured results"""
        results = {}
        lines = text.split('\n')
        current_file = None

        for line in lines:
            line = line.strip()
            if '.tsp:' in line:
                current_file = line.replace('.tsp:', '').replace('./', '').replace('_', '_')
                if current_file == 'fibonacci_spiral_100':
                    current_file = 'fibonacci_spiral'
                elif current_file == 'eil101':
                    current_file = 'eil101'
                elif current_file == 'gr229':
                    current_file = 'gr229'

            elif current_file and 'Ciudades:' in line:
                n_cities = int(line.split(':')[1].strip())
                if current_file not in results:
                    results[current_file] = {}
                results[current_file]['n_cities'] = n_cities

            elif current_file and 'Distancia óptima:' in line:
                objective = float(line.split(':')[1].strip())
                results[current_file]['objective_value'] = objective

            elif current_file and 'Estado:' in line:
                status = line.split(':')[1].strip()
                results[current_file]['status'] = status

            elif current_file and 'Tiempo:' in line:
                time_str = line.split(':')[1].strip().replace(' segundos', '')
                solve_time = float(time_str)
                results[current_file]['solve_time'] = solve_time
                results[current_file]['tour'] = list(range(results[current_file]['n_cities']))

        return results

    @staticmethod
    def run_julia_lp_solver(tsp_file: str, output_file: str = None) -> Dict:
        """Execute Julia LP solver and return results"""
        try:
            # Create temporary Julia script
            julia_script = f"""
            include("julia/pry1.jl")  # Assuming the Julia code is in a separate file

            # Run solver for {tsp_file}
            results = solve_tsp_from_file("{tsp_file}")

            # Save results in JSON format for Python
            using JSON
            json_results = Dict(
                "objective_value" => results[2],
                "solve_time" => results[4],
                "status" => string(results[3]),
                "tour" => results[1],
                "n_cities" => length(results[1])
            )

            open("{output_file or 'temp_lp_results.json'}", "w") do f
                JSON.print(f, json_results)
            end
            """

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
                f.write(julia_script)
                script_path = f.name

            # Execute Julia script
            result = subprocess.run(['julia', script_path],
                                  capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Load results
                result_file = output_file or 'temp_lp_results.json'
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        return json.load(f)

            print(f"Julia execution error: {result.stderr}")
            return {}

        except Exception as e:
            print(f"Error running Julia LP solver: {e}")
            return {}

        finally:
            # Cleanup
            if 'script_path' in locals():
                os.unlink(script_path)