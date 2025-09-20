import numpy as np
import math
from typing import List, Tuple, Dict

class TSPParser:
    def __init__(self):
        self.name = ""
        self.type = ""
        self.comment = ""
        self.dimension = 0
        self.edge_weight_type = ""
        self.coordinates = []
        self.distance_matrix = None

    def parse_file(self, filepath: str) -> Dict:
        """Parse a TSPLIB95 format file and return problem data"""
        with open(filepath, 'r') as file:
            lines = file.readlines()

        # Parse header information
        i = 0
        while i < len(lines) and not lines[i].strip().startswith('NODE_COORD_SECTION'):
            line = lines[i].strip()
            if line.startswith('NAME'):
                self.name = line.split(':')[1].strip()
            elif line.startswith('TYPE'):
                self.type = line.split(':')[1].strip()
            elif line.startswith('COMMENT'):
                self.comment = line.split(':')[1].strip()
            elif line.startswith('DIMENSION'):
                self.dimension = int(line.split(':')[1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                self.edge_weight_type = line.split(':')[1].strip()
            i += 1

        # Parse coordinates
        self.coordinates = []
        i += 1  # Skip NODE_COORD_SECTION line
        while i < len(lines) and not lines[i].strip().startswith('EOF'):
            line = lines[i].strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    self.coordinates.append((city_id, x, y))
            i += 1

        # Calculate distance matrix
        self.distance_matrix = self._calculate_distance_matrix()

        return {
            'name': self.name,
            'type': self.type,
            'comment': self.comment,
            'dimension': self.dimension,
            'edge_weight_type': self.edge_weight_type,
            'coordinates': self.coordinates,
            'distance_matrix': self.distance_matrix
        }

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix based on edge weight type"""
        n = len(self.coordinates)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.edge_weight_type == 'EUC_2D':
                        distance_matrix[i][j] = self._euclidean_distance(i, j)
                    elif self.edge_weight_type == 'GEO':
                        distance_matrix[i][j] = self._geographical_distance(i, j)
                    else:
                        raise ValueError(f"Unsupported edge weight type: {self.edge_weight_type}")

        return distance_matrix

    def _euclidean_distance(self, i: int, j: int) -> float:
        """Calculate Euclidean distance between two cities"""
        x1, y1 = self.coordinates[i][1], self.coordinates[i][2]
        x2, y2 = self.coordinates[j][1], self.coordinates[j][2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _geographical_distance(self, i: int, j: int) -> float:
        """Calculate geographical distance between two cities (in km)"""
        lat1, lon1 = self.coordinates[i][1], self.coordinates[i][2]
        lat2, lon2 = self.coordinates[j][1], self.coordinates[j][2]

        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Earth's radius in km
        R = 6371.0

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        distance = R * c
        return distance

    def get_distance(self, city1: int, city2: int) -> float:
        """Get distance between two cities (0-indexed)"""
        return self.distance_matrix[city1][city2]

    def get_num_cities(self) -> int:
        """Get number of cities"""
        return len(self.coordinates)

    def get_coordinates(self) -> List[Tuple[int, float, float]]:
        """Get list of coordinates"""
        return self.coordinates