# Análisis Comparativo: Algoritmo Genético vs Programación Lineal
## Proyecto 1 - Traveling Salesman Problem (TSP)

### Resumen Ejecutivo

Este informe presenta una comparación exhaustiva entre dos enfoques para resolver el Traveling Salesman Problem (TSP): Algoritmos Genéticos (GA) y Programación Lineal (LP).

# Comparación entre Algoritmo Genético y Programación Lineal

| Método | Escenario | Solución Encontrada | Tiempo Ejecución | % Error vs Óptimo | Observaciones |
|--------|-----------|--------------------|-----------------|--------------------|---------------|
| LP (MTZ) | eil101 | 642.09 | 154.78s | 0.00% | OPTIMAL |
| GA | eil101 | 738.08 | 10.65s | 14.95% | Heurístico |
| LP (MTZ) | gr229 | 151998.69 | 3586.62s | 0.00% | OPTIMAL |
| GA | gr229 | 206910.20 | 202.88s | 36.13% | Heurístico |
| LP (MTZ) | fibonacci_spiral | 1756.35 | 19.33s | 0.00% | OPTIMAL |
| GA | fibonacci_spiral | 2002.70 | 9.66s | 14.03% | Heurístico |


## Métricas Detalladas por Escenario

### EIL101

| Método | Tiempo Ejecución | Solución Encontrada | Solución Óptima Teórica | % Error |
|--------|------------------|--------------------|-----------------------|---------|
| LP | 154.78s | 642.09 | 642.09 | 0.00% |
| GA (mejor) | 11.29s | 738.08 | 642.09 | 14.95% |
| GA (2da mejor) | 10.33s | 751.31 | 642.09 | 17.01% |
| GA (3ra mejor) | 11.29s | 755.99 | 642.09 | 17.74% |

**Métricas del escenario:**
- Número de ciudades: 101
- Tamaño de población del GA: N/A
- Número de iteraciones del GA: 955
- Número de variables LP: 10100
- Número de restricciones LP: 10302

### GR229

| Método | Tiempo Ejecución | Solución Encontrada | Solución Óptima Teórica | % Error |
|--------|------------------|--------------------|-----------------------|---------|
| LP | 3586.62s | 151998.69 | 151998.69 | 0.00% |
| GA (mejor) | 93.50s | 206910.20 | 151998.69 | 36.13% |
| GA (2da mejor) | 92.87s | 207359.38 | 151998.69 | 36.42% |
| GA (3ra mejor) | 90.00s | 207800.51 | 151998.69 | 36.71% |

**Métricas del escenario:**
- Número de ciudades: 229
- Tamaño de población del GA: N/A
- Número de iteraciones del GA: 1853
- Número de variables LP: 52212
- Número de restricciones LP: 52670

### Fibonacci Spiral

| Método | Tiempo Ejecución | Solución Encontrada | Solución Óptima Teórica | % Error |
|--------|------------------|--------------------|-----------------------|---------|
| LP | 19.33s | 1756.35 | 1756.35 | 0.00% |
| GA (mejor) | 10.19s | 2002.70 | 1756.35 | 14.03% |
| GA (2da mejor) | 10.08s | 2013.35 | 1756.35 | 14.63% |
| GA (3ra mejor) | 11.19s | 2052.76 | 1756.35 | 16.88% |

**Métricas del escenario:**
- Número de ciudades: 100
- Tamaño de población del GA: N/A
- Número de iteraciones del GA: 861
- Número de variables LP: 9900
- Número de restricciones LP: 10100


### Análisis de Métodos

#### Programación Lineal (Formulación MTZ)
- **Ventajas:** Garantiza solución óptima, formulación matemática exacta
- **Desventajas:** Tiempo exponencial para problemas grandes, uso intensivo de memoria
- **Aplicabilidad:** Ideal para problemas pequeños (<150 ciudades)

#### Algoritmo Genético
- **Ventajas:** Escalable, tiempo de ejecución predecible, buena calidad de soluciones
- **Desventajas:** No garantiza optimalidad, requiere ajuste de parámetros
- **Aplicabilidad:** Excelente para problemas grandes (>100 ciudades)

### Conclusiones

1. **Para problemas pequeños (≤101 ciudades):** LP encuentra soluciones óptimas en tiempo razonable
2. **Para problemas medianos (100-300 ciudades):** GA ofrece mejor balance tiempo/calidad
3. **Para problemas grandes (>300 ciudades):** GA es la única opción práctica

### Implementación Técnica

- **LP:** Julia + JuMP + HiGHS solver, formulación Miller-Tucker-Zemlin
- **GA:** Python, selección por torneo, cruce OX, mutación 2-opt
- **Visualización:** Matplotlib para convergencia y tours, integración tiempo real

---
*Generado automáticamente por el sistema de análisis TSP*
