# Comparativa de Hill Climbing vs Simulated Annealing

Aplicación en Python con Streamlit para comparar los algoritmos `Hill Climbing` y `Simulated Annealing (SA)` sobre dos problemas de optimización:

- `TSP (Traveling Salesman Problem / Viajero)`
- `N-Reinas`

El proyecto permite:

- ejecutar una comparación visual en una sola instancia
- medir rendimiento con distintos tamaños de problema `n`
- analizar el efecto de los hiperparámetros de `Simulated Annealing`

## Objetivo

Este proyecto fue desarrollado como apoyo para una tarea académica de Inteligencia Artificial, enfocada en:

- implementar `Simulated Annealing` en Python
- comparar `Hill Climbing` vs `Simulated Annealing`
- evaluar calidad de solución y tiempo de ejecución
- estudiar el impacto de los hiperparámetros de SA

## Tecnologías usadas

- Python 3.10+
- Streamlit
- NumPy
- Pandas
- Matplotlib

## Estructura del proyecto

```text
Practica Python/
├── app.py
├── requirement.txt
├── core/
│   ├── algorithms.py
│   └── problems.py
└── .gitignore
```

Descripción de archivos:

- `app.py`: interfaz principal en Streamlit y lógica de experimentación
- `core/algorithms.py`: implementación de `Hill Climbing` y `Simulated Annealing`
- `core/problems.py`: definición de los problemas `TSP` y `N-Reinas`
- `requirement.txt`: dependencias del proyecto

## Instalación

### 1. Clonar o descargar el proyecto

Si usas Git:

```bash
git clone <URL_DEL_REPOSITORIO>
cd "Practica Python"
```

Si ya tienes la carpeta local, entra directamente al proyecto.

### 2. Crear un entorno virtual

En Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

En Windows CMD:

```cmd
python -m venv venv
venv\Scripts\activate
```

En Linux o macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirement.txt
```

Nota:

- el archivo de dependencias se llama `requirement.txt`
- si prefieres seguir la convención estándar, puedes renombrarlo a `requirements.txt`

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

Luego abre en el navegador la URL local que Streamlit muestre en consola, normalmente:

```text
http://localhost:8501
```

## Problemas implementados

### TSP

El problema del viajero consiste en encontrar una ruta que recorra todas las ciudades y regrese al punto inicial minimizando la distancia total.

En este proyecto:

- las ciudades se generan aleatoriamente en un plano 2D
- un estado es una permutación de ciudades
- el `fitness` es la distancia total de la ruta
- menor `fitness` significa mejor solución

### N-Reinas

El problema de N-Reinas busca ubicar `n` reinas en un tablero de `n x n` sin que se ataquen entre sí.

En este proyecto:

- un estado indica la fila donde se coloca la reina de cada columna
- el `fitness` es el número de conflictos entre reinas
- el óptimo ideal es `fitness = 0`

## Algoritmos implementados

### Hill Climbing

Estrategia de búsqueda local que:

- parte de una solución inicial
- genera todos los vecinos
- elige el vecino con mejor `fitness`
- se detiene cuando no encuentra mejora

Ventajas:

- simple
- rápido en problemas pequeños o medianos
- suele converger en pocas iteraciones

Desventajas:

- puede quedarse atrapado en óptimos locales

### Simulated Annealing

Variante probabilística de búsqueda local que:

- parte de una solución inicial
- explora vecinos aleatorios
- acepta mejoras directas
- en ocasiones acepta soluciones peores con cierta probabilidad
- reduce gradualmente esa probabilidad usando una temperatura

Ventajas:

- puede escapar de óptimos locales
- es útil en paisajes de búsqueda complejos

Desventajas:

- depende bastante de sus hiperparámetros
- puede ser más lento o menos estable si está mal configurado

## Parámetros de Simulated Annealing

En la aplicación se pueden configurar:

- `T0`: temperatura inicial
- `alpha`: factor de enfriamiento
- `Tmin`: temperatura mínima o final

Interpretación:

- `T0` alta: más exploración al inicio
- `alpha` cercano a `1`: enfriamiento más lento, más iteraciones
- `Tmin` pequeña: refinamiento más largo antes de detenerse

Trade-off general:

- más exploración suele mejorar la calidad final
- pero incrementa el tiempo de ejecución

## Qué muestra la aplicación

La app tiene tres pestañas principales.

### 1. Ejecución individual

Permite comparar ambos algoritmos sobre una sola instancia del problema.

Qué muestra:

- visualización del estado actual
- tabla final con `fitness` y tiempo de ambos algoritmos
- gráfico de convergencia

Para qué sirve:

- entender visualmente el comportamiento de cada algoritmo
- mostrar una corrida concreta en la exposición

### 2. Benchmark por n

Permite evaluar ambos algoritmos para distintos tamaños de problema.

Entrada:

- lista de valores de `n`, por ejemplo `4, 8, 12, 16, 20`
- número de corridas por cada `n`

Qué hace:

- ejecuta `Hill Climbing` y `Simulated Annealing` varias veces
- usa la misma semilla base y derivaciones reproducibles
- registra `fitness`, tiempo y ganador por corrida

Qué muestra:

- tabla detallada por corrida
- resumen agregado por cada valor de `n`
- gráfico de `fitness` promedio por tamaño
- gráfico de tiempo promedio por tamaño

Para qué sirve:

- ver cuál algoritmo se comporta mejor al aumentar la complejidad
- comparar calidad de solución contra costo computacional

### 3. Análisis de hiperparámetros

Permite estudiar cómo cambian los resultados de `Simulated Annealing` según `T0`, `alpha` y `Tmin`.

Entrada:

- un valor de `n`
- número de corridas por configuración
- listas de valores para `T0`, `alpha` y `Tmin`

Qué hace:

- genera una grilla de combinaciones
- ejecuta SA para cada combinación
- compara cada configuración contra un baseline de `Hill Climbing`

Qué muestra:

- baseline de `Hill Climbing`
- resultados agregados de SA
- top de mejores configuraciones
- ranking de calidad
- ranking de tiempo
- efecto individual de `T0`
- efecto individual de `alpha`
- efecto individual de `Tmin`

Para qué sirve:

- justificar qué parámetros funcionan mejor
- mostrar experimentalmente el efecto de la exploración y el enfriamiento

## Semilla base

La `semilla base` controla la aleatoriedad del experimento.

Se usa para:

- generar instancias reproducibles del problema
- repetir la misma comparación entre algoritmos
- obtener resultados consistentes entre ejecuciones

Si se mantiene la misma semilla:

- el experimento puede repetirse bajo las mismas condiciones

Si se cambia la semilla:

- cambia la instancia aleatoria del problema
- pueden cambiar los resultados

## Cómo interpretar los resultados

### Fitness

- en `TSP`, un menor valor significa una ruta más corta
- en `N-Reinas`, un menor valor significa menos conflictos
- en `N-Reinas`, `0` significa solución perfecta

### Tiempo

- representa el costo computacional de cada algoritmo
- no siempre el mejor `fitness` implica el mejor tiempo

### Ganador

En el benchmark, el ganador de cada corrida se define por el menor `fitness` final:

- gana `Hill Climbing` si su `fitness` es menor
- gana `Simulated Annealing` si su `fitness` es menor
- hay empate si ambos llegan al mismo valor

## Ejemplos de uso

### Comparación rápida

- problema: `N-Reinas`
- `n = 15`
- `T0 = 100`
- `alpha = 0.95`
- `Tmin = 0.1`

Sirve para observar una corrida individual y el gráfico de convergencia.

### Benchmark recomendado

- problema: `N-Reinas`
- valores de `n`: `4, 8, 12, 16, 20, 24, 28, 30`
- corridas: `3` o `5`

Sirve para obtener tablas y gráficas para la presentación.

### Barrido inicial de hiperparámetros

- `T0`: `10, 50, 100, 200`
- `alpha`: `0.85, 0.90, 0.95, 0.99`
- `Tmin`: `1, 0.1, 0.01`

Sirve para identificar configuraciones razonables de SA.

## Limitaciones actuales

- `Hill Climbing` evalúa todos los vecinos, mientras que `Simulated Annealing` evalúa un vecino aleatorio por paso
- por eso, en algunos experimentos `Hill Climbing` puede verse muy fuerte en calidad final
- los resultados dependen bastante de los hiperparámetros elegidos para SA
- en tamaños grandes, SA puede necesitar un enfriamiento más lento para competir mejor

## Posibles mejoras futuras

- agregar más iteraciones por cada temperatura en SA
- permitir reinicios aleatorios para Hill Climbing
- exportar resultados a CSV
- guardar gráficas automáticamente
- añadir más problemas de optimización

## Créditos

Proyecto académico para la comparación de algoritmos de búsqueda local en Inteligencia Artificial.

## Licencia

Este proyecto puede adaptarse según las necesidades del curso o del grupo de trabajo. Si el docente o la institución requieren una licencia específica, agréguela aquí.
