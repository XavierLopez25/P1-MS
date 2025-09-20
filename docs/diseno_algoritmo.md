# Diseño del Algoritmo Genético para TSP

## Resumen Ejecutivo

Este documento describe el diseño e implementación de un algoritmo genético (GA) para resolver el Traveling Salesman Problem (TSP). El algoritmo ha sido desarrollado como parte del Proyecto 1 de Métodos de Solución, enfocándose en la comparación entre metaheurísticas y programación lineal.

## 1. Introducción

### 1.1 Problema del Viajante (TSP)
El TSP es un problema de optimización combinatoria que busca encontrar el tour más corto que visite todas las ciudades exactamente una vez y regrese al punto de origen. Formalmente:

- **Entrada**: Conjunto de n ciudades y matriz de distancias D[i][j]
- **Objetivo**: Minimizar la distancia total del tour
- **Restricciones**: Cada ciudad debe ser visitada exactamente una vez

### 1.2 Justificación del Algoritmo Genético
Los algoritmos genéticos son especialmente adecuados para el TSP porque:
- Mantienen múltiples soluciones simultáneamente (diversidad)
- Pueden escapar de óptimos locales
- Son robustos ante diferentes tipos de instancia
- Proporcionan soluciones de buena calidad en tiempo razonable

## 2. Arquitectura del Algoritmo

### 2.1 Representación
**Tipo**: Permutación directa
**Descripción**: Cada individuo se representa como una lista de enteros [0, 1, 2, ..., n-1] donde cada número representa una ciudad y el orden indica la secuencia de visita.

**Ejemplo**: Para 5 ciudades, [2, 0, 4, 1, 3] representa el tour 2→0→4→1→3→2

**Ventajas**:
- Representación intuitiva y directa
- Garantiza que cada ciudad aparezca exactamente una vez
- Facilita la implementación de operadores especializados

### 2.2 Función de Fitness
```python
def fitness(tour, distance_matrix):
    total_distance = 0
    for i in range(len(tour)):
        current = tour[i]
        next_city = tour[(i + 1) % len(tour)]
        total_distance += distance_matrix[current][next_city]
    return total_distance
```

**Características**:
- Minimización (menor distancia = mejor fitness)
- Incluye el retorno a la ciudad de origen
- Soporta matrices de distancia euclidianas y geográficas

## 3. Operadores Genéticos

### 3.1 Operador de Selección: Torneo
**Tipo**: Torneo binario/ternario
**Parámetros**:
- `tournament_size`: Número de individuos en cada torneo (default: 3)

**Algoritmo**:
1. Seleccionar aleatoriamente `tournament_size` individuos
2. Retornar el individuo con mejor fitness (menor distancia)

**Ventajas**:
- Presión de selección ajustable
- Eficiente computacionalmente
- Mantiene diversidad poblacional

### 3.2 Operador de Cruce: Order Crossover (OX)
**Justificación**: OX preserva las subsequencias relativas de ciudades de ambos padres, crucial para TSP.

**Algoritmo**:
1. Seleccionar dos puntos de corte aleatorios
2. Copiar segmento del padre 1 al offspring 1
3. Llenar posiciones restantes con ciudades del padre 2 en orden, omitiendo duplicados
4. Repetir proceso invertido para offspring 2

**Ejemplo**:
```
Padre 1: [1, 2, 3, 4, 5, 6, 7, 8]
Padre 2: [3, 7, 5, 1, 6, 8, 2, 4]
Puntos:     ^     ^
Offspring: [7, 5, 3, 4, 5, 1, 6, 8]
```

### 3.3 Operador de Mutación: 2-opt
**Justificación**: 2-opt es específico para TSP y mejora la calidad local de tours.

**Algoritmo**:
1. Seleccionar dos aristas aleatorias del tour
2. Reconectar eliminando cruces (reversión del segmento)
3. Aplicar con probabilidad `mutation_rate`

**Ejemplo**:
```
Original: [1, 2, 3, 4, 5, 6]
Aristas seleccionadas: (2,3) y (5,6)
Resultado: [1, 2, 5, 4, 3, 6]  // Segmento [3,4,5] revertido
```

### 3.4 Estrategia de Reemplazo: Elitismo
**Parámetros**:
- `elite_size`: Número de mejores individuos preservados (default: 10% población)

**Beneficios**:
- Garantiza que las mejores soluciones no se pierdan
- Acelera la convergencia
- Mantiene calidad de soluciones a lo largo de generaciones

## 4. Parámetros del Algoritmo

### 4.1 Parámetros Principales
| Parámetro | Valor Default | Rango Recomendado | Descripción |
|-----------|---------------|-------------------|-------------|
| `population_size` | 100 | 50-200 | Tamaño de la población |
| `max_generations` | 1000 | 500-2000 | Número máximo de generaciones |
| `elite_size` | 10 | 5-20% población | Individuos elite preservados |
| `crossover_rate` | 0.8 | 0.6-0.9 | Probabilidad de cruce |
| `mutation_rate` | 0.02 | 0.01-0.05 | Probabilidad de mutación |
| `tournament_size` | 3 | 2-5 | Tamaño del torneo |

### 4.2 Parámetros Adaptativos
- **Población**: Escalada según número de ciudades (min 50, max 200)
- **Generaciones**: Ajustadas según complejidad del problema
- **Convergencia**: Parada temprana tras 50-100 generaciones sin mejora

## 5. Estrategias de Optimización

### 5.1 Inicialización de Población
- **Método**: Permutaciones aleatorias uniformes
- **Ventaja**: Cobertura completa del espacio de búsqueda inicial
- **Implementación**: Shuffle de tour base [0,1,2,...,n-1]

### 5.2 Criterio de Parada
**Condiciones múltiples**:
1. Número máximo de generaciones alcanzado
2. Convergencia: No mejora por N generaciones consecutivas
3. Tiempo límite (opcional)

### 5.3 Mantenimiento de Diversidad
- Torneo probabilístico evita convergencia prematura
- Mutación 2-opt introduce variabilidad local
- Elite size controlado para equilibrar exploración/explotación

## 6. Implementación Técnica

### 6.1 Estructura de Clases
```
GeneticAlgorithmTSP
├── __init__()          # Configuración de parámetros
├── run()               # Ciclo principal del algoritmo
├── multiple_runs()     # Ejecuciones múltiples para estadísticas
└── _create_next_generation()  # Aplicación de operadores

GeneticOperators
├── tournament_selection()
├── order_crossover()
├── two_opt_mutation()
├── calculate_fitness()
└── create_initial_population()
```

### 6.2 Manejo de Datos
**Parser TSP**:
- Soporte para formatos TSPLIB95
- Tipos de distancia: EUC_2D (Euclidiana) y GEO (Geográfica)
- Cálculo automático de matriz de distancias

**Escenarios soportados**:
1. **EIL101**: 101 ciudades, distancias euclidianas
2. **GR229**: 229 ciudades, distancias geográficas
3. **Fibonacci Spiral**: 100 ciudades en patrón espiral áureo

## 7. Análisis de Complejidad

### 7.1 Complejidad Temporal
- **Por generación**: O(P × N²) donde P = población, N = ciudades
- **Evaluación fitness**: O(N) por individuo
- **Selección**: O(P × log P) para torneo
- **Cruce OX**: O(N) por pareja
- **Mutación 2-opt**: O(1) en promedio

### 7.2 Complejidad Espacial
- **Población**: O(P × N)
- **Matriz distancias**: O(N²)
- **Total**: O(P × N + N²)

## 8. Resultados Experimentales Preliminares

### 8.1 Configuración de Pruebas
- 3 ejecuciones independientes por escenario
- Parámetros estándar del algoritmo
- Criterio de parada: 200-300 generaciones máx.

### 8.2 Resultados Obtenidos
| Escenario | Mejor Solución | Tiempo Promedio | Generaciones | Desv. Estándar |
|-----------|----------------|-----------------|--------------|----------------|
| EIL101 | 1220.82 | 1.44s | 200 | 67.08 |
| GR229 | 400341.07 | 13.00s | 300 | 15123.33 |
| Fibonacci Spiral | 3031.71 | 2.40s | 250 | 46.69 |

### 8.3 Análisis de Rendimiento
- **EIL101**: Convergencia rápida, buena consistencia
- **GR229**: Mayor variabilidad debido a escala geográfica
- **Fibonacci**: Excelente convergencia en patrón estructurado

## 9. Conclusiones

### 9.1 Fortalezas del Diseño
1. **Robustez**: Funciona efectivamente en diferentes tipos de instancia
2. **Flexibilidad**: Parámetros configurables según problema
3. **Eficiencia**: Balance adecuado entre calidad y tiempo de ejecución
4. **Escalabilidad**: Maneja problemas de 50-300 ciudades efectivamente

### 9.2 Limitaciones Identificadas
1. **Convergencia prematura**: En problemas altamente estructurados
2. **Sensibilidad paramétrica**: Requiere ajuste fino para instancias específicas
3. **Memoria**: O(N²) puede ser limitante para problemas muy grandes

### 9.3 Posibles Mejoras
1. **Hibridización**: Combinar con búsqueda local (2-opt completo)
2. **Mutación adaptativa**: Ajustar tasa según diversidad poblacional
3. **Selección adaptativa**: Variar presión selectiva dinámicamente
4. **Paralelización**: Evaluación paralela de fitness

## 10. Referencias Técnicas

### 10.1 Operadores Implementados
- **Order Crossover**: Davis, L. (1985)
- **2-opt Mutation**: Lin, S. & Kernighan, B. (1973)
- **Tournament Selection**: Miller & Goldberg (1995)

### 10.2 Benchmarks Utilizados
- **TSPLIB95**: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- **EIL101**: Christofides & Eilon benchmark
- **GR229**: Groetschel Asia/Australia subproblem

---

**Autor**: [Nombre del estudiante]
**Proyecto**: Algoritmos Genéticos + LP para TSP
**Fecha**: Octubre 2024
**Versión**: 1.0