# Algoritmo Genético para TSP - Proyecto 1

## Descripción

Implementación de un algoritmo genético para resolver el Traveling Salesman Problem (TSP) como parte del Proyecto 1 de Métodos de Solución. El proyecto incluye pruebas en tres escenarios diferentes y comparación con programación lineal.

## Estructura del Proyecto

```
P1-MS/
├── src/                          # Código fuente
│   ├── tsp_parser.py            # Parser para archivos TSPLIB95
│   ├── genetic_algorithm.py     # Algoritmo genético principal
│   ├── operators.py             # Operadores genéticos
│   ├── custom_scenario.py       # Generador de escenario personalizado
│   ├── utils.py                 # Utilidades y visualización
│   └── main.py                  # Script principal
├── docs/                        # Documentación
│   └── diseño_algoritmo.md      # Diseño detallado del algoritmo
├── results/                     # Resultados de experimentos
├── eil101.tsp                   # Dataset TSPLIB95 (101 ciudades)
├── gr229.tsp                    # Dataset TSPLIB95 (229 ciudades)
├── fibonacci_spiral_100.tsp     # Escenario personalizado generado
├── test_quick.py                # Prueba rápida del sistema
└── run_experiments.py           # Experimentos de demostración
```

## Requisitos

```bash
pip install numpy matplotlib
```

## Uso Rápido

### 1. Prueba de funcionamiento

```bash
python test_quick.py
```

### 2. Experimentos de demostración

```bash
python run_experiments.py
```

### 3. Sistema completo interactivo

```bash
cd src
python main.py
```

## Escenarios de Prueba

### 1. EIL101

- **Fuente**: TSPLIB95 (Christofides/Eilon)
- **Ciudades**: 101
- **Tipo**: Distancias euclidianas (EUC_2D)
- **Características**: Problema clásico de benchmark

### 2. GR229

- **Fuente**: TSPLIB95 (Groetschel)
- **Ciudades**: 229
- **Tipo**: Distancias geográficas (GEO)
- **Características**: Subproblema Asia/Australia

### 3. Fibonacci Spiral (Personalizado)

- **Fuente**: Generado automáticamente
- **Ciudades**: 100
- **Tipo**: Distancias euclidianas
- **Características**: Patrón espiral basado en proporción áurea

## Configuración del Algoritmo

### Parámetros Principales

- **Población**: 50-200 individuos (adaptativo según problema)
- **Generaciones**: 500-2000 (con parada temprana)
- **Selección**: Torneo (tamaño 3)
- **Cruce**: Order Crossover (OX) - 80%
- **Mutación**: 2-opt - 2%
- **Elitismo**: 10% de la población

### Operadores Genéticos

- **Representación**: Permutación directa de ciudades
- **Fitness**: Distancia total del tour (minimización)
- **Selección**: Torneo probabilístico
- **Cruce**: Order Crossover (preserva subsequencias)
- **Mutación**: 2-opt (mejora local)

## Resultados de Ejemplo

| Escenario        | Mejor Solución | Tiempo Promedio | Generaciones |
| ---------------- | -------------- | --------------- | ------------ |
| EIL101           | 1220.82        | 1.44s           | 200          |
| GR229            | 400341.07      | 13.00s          | 300          |
| Fibonacci Spiral | 3031.71        | 2.40s           | 250          |

## Archivos de Salida

### Resultados JSON

- `results/eil101_results.json`
- `results/gr229_results.json`
- `results/custom_results.json`

### Visualizaciones (si matplotlib disponible)

- Gráficos de convergencia
- Visualización de mejores tours
- Comparaciones entre escenarios

### Reportes

- `results/comparison_table.md`: Tabla comparativa
- `docs/diseño_algoritmo.md`: Documentación técnica

## API Principal

### Uso Básico

```python
from src.tsp_parser import TSPParser
from src.genetic_algorithm import GeneticAlgorithmTSP

# Cargar problema
parser = TSPParser()
data = parser.parse_file('eil101.tsp')

# Configurar algoritmo
ga = GeneticAlgorithmTSP(
    distance_matrix=data['distance_matrix'],
    population_size=100,
    max_generations=500
)

# Ejecutar
result = ga.run()
print(f"Mejor solución: {result['best_fitness']}")
```

### Ejecuciones Múltiples

```python
# Para estadísticas robustas
results = ga.multiple_runs(num_runs=10)
print(f"Promedio: {results['mean_fitness']}")
print(f"Desviación: {results['std_fitness']}")
```

## Personalización

### Crear Nuevo Escenario

```python
from src.custom_scenario import create_fibonacci_spiral_scenario

# Generar escenario personalizado
scenario = create_fibonacci_spiral_scenario(
    num_cities=150,
    output_file="mi_escenario.tsp"
)
```

### Ajustar Parámetros

```python
ga = GeneticAlgorithmTSP(
    distance_matrix=data['distance_matrix'],
    population_size=200,      # Población más grande
    mutation_rate=0.05,       # Más mutación
    crossover_rate=0.9,       # Más cruce
    tournament_size=5,        # Torneo más selectivo
    mutation_type='swap'      # Tipo de mutación alternativo
)
```

## Validación y Testing

El sistema incluye validaciones automáticas:

- Verificación de tours válidos (todas las ciudades visitadas)
- Cálculo correcto de distancias
- Consistencia de operadores genéticos

## Limitaciones Conocidas

1. **Memoria**: O(N²) para matriz de distancias
2. **Tiempo**: Escalabilidad limitada para >500 ciudades
3. **Convergencia**: Puede converger prematuramente en algunos casos

## Troubleshooting

### Error: No module found

```bash
# Verificar de estar en el directorio correcto
cd P1-MS
python -c "import sys; print(sys.path)"
```

### Matplotlib no disponible

- Las funciones de visualización son opcionales
- El algoritmo funciona sin matplotlib
- Instalar con: `pip install matplotlib`

### Memoria insuficiente

- Reducir `population_size`
- Usar menos `num_runs`
- Para problemas >300 ciudades, considerar implementación optimizada

## Contribución

Este proyecto es parte de una tarea académica. Para mejoras:

1. Fork del repositorio
2. Crear branch para nueva feature
3. Implementar mejora
4. Documentar cambios
5. Pull request con descripción detallada

## Autor

**Proyecto**: Algoritmos Genéticos + LP para TSP
**Curso**: Métodos de Solución
**Fecha**: Octubre 2024

## Referencias

- TSPLIB95: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- Genetic Algorithms for TSP: Goldberg, D.E. (1989)
- Order Crossover: Davis, L. (1985)
- 2-opt Heuristic: Lin, S. & Kernighan, B. (1973)
