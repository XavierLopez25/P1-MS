import numpy as np
import math
from typing import List, Tuple

class FibonacciSpiralGenerator:
    def __init__(self):
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.golden_angle = 2 * math.pi / (self.golden_ratio ** 2)

    def generate_cities(self, num_cities: int = 100, scale: float = 10.0) -> List[Tuple[int, float, float]]:
        """
        Generate cities arranged in a Fibonacci spiral pattern

        Args:
            num_cities: Number of cities to generate (50-200)
            scale: Scale factor for coordinates

        Returns:
            List of tuples (city_id, x, y)
        """
        if num_cities < 50 or num_cities > 200:
            raise ValueError("Number of cities must be between 50 and 200")

        cities = []

        for i in range(num_cities):
            # Calculate radius using square root for even distribution
            radius = scale * math.sqrt(i + 1)

            # Calculate angle using golden angle
            angle = i * self.golden_angle

            # Convert polar to cartesian coordinates
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            # Add some controlled randomness to make it more realistic
            noise_factor = 0.1 * scale
            x += np.random.uniform(-noise_factor, noise_factor)
            y += np.random.uniform(-noise_factor, noise_factor)

            cities.append((i + 1, round(x, 2), round(y, 2)))

        return cities

    def save_tsp_file(self, cities: List[Tuple[int, float, float]], filename: str):
        """Save cities in TSPLIB95 format"""
        with open(filename, 'w') as f:
            f.write(f"NAME: fibonacci_spiral_{len(cities)}\n")
            f.write("TYPE: TSP\n")
            f.write(f"COMMENT: {len(cities)}-city problem arranged in Fibonacci spiral pattern\n")
            f.write(f"DIMENSION: {len(cities)}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")

            for city_id, x, y in cities:
                f.write(f"{city_id} {x} {y}\n")

            f.write("EOF\n")

    def calculate_distance_matrix(self, cities: List[Tuple[int, float, float]]) -> np.ndarray:
        """Calculate Euclidean distance matrix for the cities"""
        n = len(cities)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = cities[i][1], cities[i][2]
                    x2, y2 = cities[j][1], cities[j][2]
                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    distance_matrix[i][j] = distance

        return distance_matrix

def create_fibonacci_spiral_scenario(num_cities: int = 100, output_file: str = None):
    """
    Create a Fibonacci spiral TSP scenario

    Args:
        num_cities: Number of cities (default 100)
        output_file: Output filename (if None, uses default name)

    Returns:
        Dictionary with scenario data
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    generator = FibonacciSpiralGenerator()
    cities = generator.generate_cities(num_cities)
    distance_matrix = generator.calculate_distance_matrix(cities)

    # Save to file if specified
    if output_file:
        generator.save_tsp_file(cities, output_file)

    return {
        'name': f'fibonacci_spiral_{num_cities}',
        'type': 'TSP',
        'comment': f'{num_cities}-city problem arranged in Fibonacci spiral pattern',
        'dimension': num_cities,
        'edge_weight_type': 'EUC_2D',
        'coordinates': cities,
        'distance_matrix': distance_matrix
    }

if __name__ == "__main__":
    # Generate the custom scenario
    scenario = create_fibonacci_spiral_scenario(100, "../fibonacci_spiral_100.tsp")
    print(f"Generated Fibonacci spiral scenario with {scenario['dimension']} cities")
    print(f"Pattern: {scenario['comment']}")