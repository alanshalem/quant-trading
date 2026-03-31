#!/usr/bin/env python
# coding: utf-8

# # Módulo 02: Arrays y Estructuras de Datos
#
# **Quant Trading Accelerator**
#
# ---

# ## Tabla de Contenidos
#
# 1. [Objetivos de Aprendizaje](#learning-objectives)
# 2. [Introducción a los Arrays](#introduction-to-arrays)
# 3. [Listas de Python](#python-lists)
# 4. [Operaciones con Listas](#list-operations)
# 5. [Arrays de NumPy](#numpy-arrays)
# 6. [Logaritmos en Finanzas](#logarithms-in-finance)
# 7. [Log Returns](#log-returns)
# 8. [Ejercicios Prácticos](#practical-exercises)
# 9. [Puntos Clave](#key-takeaways)

# ---
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este módulo, serás capaz de:
#
# - Entender los arrays como el bloque fundamental del trading cuantitativo
# - Trabajar con listas de Python y arrays de NumPy de forma eficiente
# - Entender la complejidad computacional de las operaciones con listas
# - Aplicar logaritmos a cálculos financieros
# - Calcular e interpretar log returns
# - Entender por qué los log returns son preferidos en finanzas cuantitativas

# ---
#
# ## Introducción a los Arrays
#
# Los arrays son **la estructura de datos más fundamental** en trading cuantitativo, machine learning e IA.
# Cada serie de precios, cada vector de features y cada parámetro de modelo se almacena como un array.
#
# En trading, trabajamos constantemente con secuencias de datos:
# - Series temporales de precios: `[100.5, 101.2, 99.8, 102.1, ...]`
# - Historial de PnL por operación: `[150.0, -75.0, 200.0, -50.0, ...]`
# - Features del modelo: `[lag_1, lag_2, volatility, momentum, ...]`

# ---
#
# ## Listas de Python
#
# Las listas de Python son la estructura tipo array más simple. Son flexibles pero no están optimizadas para cálculos numéricos.
#
# ### Crear una Lista
#
# Creemos una serie de precios simple:

prices = [10.2, 9.4, 9.9, 10.5]
prices

# Verificar el tipo
type(prices)

# Obtener la longitud (número de elementos)
len(prices)

# ### Acceder a Elementos
#
# Python usa **indexación basada en cero** (el primer elemento está en el índice 0):
#
# ```
# Índice:   0      1      2      3
# Precios: [10.2,  9.4,   9.9,  10.5]
# Neg:      -4     -3     -2     -1
# ```

# Indexación hacia adelante (0, 1, 2, ...)
print(f"prices[0] = {prices[0]}")  # Primer elemento
print(f"prices[1] = {prices[1]}")  # Segundo elemento
print(f"prices[2] = {prices[2]}")  # Tercer elemento
print(f"prices[3] = {prices[3]}")  # Cuarto elemento

# Indexación negativa (-1, -2, -3, ...)
print(f"prices[-1] = {prices[-1]}")  # Último elemento
print(f"prices[-2] = {prices[-2]}")  # Penúltimo
print(f"prices[-3] = {prices[-3]}")  # Antepenúltimo
print(f"prices[-4] = {prices[-4]}")  # Cuarto desde el final (igual que el primero)

# Acceder fuera de los límites lanza un IndexError
# Descomenta para ver el error:
# prices[5]

# ---
#
# ## Operaciones con Listas
#
# ### Actualizar Elementos

prices = [10.2, 9.4, 9.9, 10.5]
prices[0] = None  # Reemplazar primer elemento
prices

prices[-1] = None  # Reemplazar último elemento
prices

# ### Eliminar Elementos
#
# Hay múltiples formas de eliminar elementos de una lista:

prices = [10.2, 9.4, 9.9, 10.5]

# pop() elimina y retorna el último elemento
last_price = prices.pop()
print(f"Eliminado: {last_price}, Restantes: {prices}")

prices = [10.2, 9.4, 9.9, 10.5]

# pop(0) elimina y retorna el primer elemento
first_price = prices.pop(0)
print(f"Eliminado: {first_price}, Restantes: {prices}")

prices = [10.2, 9.4, 9.9]

# del elimina por índice sin retornar
del prices[0]
prices

# ### Consideración de Rendimiento: O(1) vs O(n)
#
# **Crítico para sistemas de trading**: Las operaciones de lista tienen diferentes complejidades computacionales.
#
# - `pop()` (eliminar último): **O(1)** - Tiempo constante
# - `pop(0)` (eliminar primero): **O(n)** - Tiempo lineal (debe desplazar todos los elementos)
#
# Para conjuntos de datos grandes, esta diferencia es significativa:

import time

n = 200_000_000
prices_ts = [1.0 for _ in range(n)]

# Eliminar desde el frente - O(n) - ¡LENTO!
start = time.time()
prices_ts.pop(0)
elapsed = time.time() - start
print(f"pop(0) time: {elapsed*1000:.1f} ms")

prices_ts = [1.0 for _ in range(n)]

# Eliminar desde el final - O(1) - ¡RÁPIDO!
start = time.time()
prices_ts.pop()
elapsed = time.time() - start
print(f"pop() time: {elapsed*1000:.4f} ms")

# **Punto Clave**: Al construir sistemas de trading que procesan millones de datos,
# siempre considera la complejidad computacional de tus operaciones.
#
# ### Agregar Elementos

prices = []

# append() agrega un solo elemento al final
prices.append(10.5)
prices

# extend() agrega múltiples elementos
prices.extend([11.4, 9.5, 12.3])
prices

# ### Arrays Homogéneos vs Inhomogéneos
#
# Las listas de Python pueden contener tipos mixtos (inhomogéneas):

# Inhomogéneo - diferentes tipos (evitar en computación numérica)
mixed = [1.0, "a", True, 2]
mixed

# Homogéneo - mismo tipo (preferido para computación numérica)
floats = [1.0, 2.0, 3.0, 4.0]
integers = [1, 2, 3, 4]
print(f"Floats: {floats}")
print(f"Integers: {integers}")

# **Buena Práctica**: Siempre usa arrays homogéneos para datos numéricos para evitar sobrecarga de conversión de tipos.
#
# ---
#
# ## Bucles
#
# Los bucles nos permiten iterar sobre arrays y realizar cálculos:

# Bucle range básico
for i in range(5):
    print(i)

# Iterar sobre una lista
prices = [10.2, 9.5, 11.5, 9.4]
for price in prices:
    print(price)

# ### Calcular PnL Total con un Bucle
#
# Una tarea común en trading - sumar PnLs individuales de operaciones:

trade_pnls = [1.2, -2.0, -1.0, 4.1]

total_pnl = 0.0
for trade_pnl in trade_pnls:
    total_pnl += trade_pnl

print(f"Total P&L: ${total_pnl:.2f}")

# Verificar: cálculo manual
1.2 + (-2.0) + (-1.0) + 4.1

# ---
#
# ## Arrays de NumPy
#
# **NumPy** (Numerical Python) es la base de la computación científica en Python.
# Los arrays de NumPy son:
# - Más rápidos que las listas de Python (implementados en C)
# - Soportan operaciones vectorizadas (no se necesitan bucles explícitos)
# - Eficientes en memoria (disposición contigua en memoria)

import numpy as np

# Crear un array de 100 millones de unos
n = 100_000_000
a = np.ones(n)
a

len(a)

# ### Operaciones Vectorizadas vs Bucles
#
# Las operaciones vectorizadas de NumPy son **órdenes de magnitud más rápidas** que los bucles de Python:

# Suma NumPy - vectorizada (usa instrucciones SIMD)
import time

start = time.time()
result = np.sum(a)
elapsed = time.time() - start
print(f"NumPy sum: {result:.0f}")
print(f"Time: {elapsed*1000:.1f} ms")

# Bucle Python - secuencial (interpretado)
start = time.time()
total = 0.0
for val in a:
    total += val
elapsed = time.time() - start
print(f"Loop sum: {total:.0f}")
print(f"Time: {elapsed*1000:.1f} ms")

# **¿Por qué NumPy es más rápido?**
#
# 1. **SIMD (Single Instruction, Multiple Data)**: Procesa múltiples valores en una sola instrucción de CPU
# 2. **Sin verificación de tipos**: Todos los elementos son del mismo tipo
# 3. **Localidad de memoria**: Acceso contiguo a memoria
# 4. **Código C compilado**: Sin sobrecarga del intérprete de Python
#
# ---
#
# ## Logaritmos en Finanzas
#
# Los logaritmos son fundamentales en finanzas cuantitativas. El logaritmo natural (ln) es la inversa de la función exponencial:
#
# $$e^x = y \iff \ln(y) = x$$
#
# Donde $e \approx 2.71828$ es el número de Euler.

# Función exponencial
np.exp(2) # e^2 = 7.38905609893065

# Log es la inversa de exp
np.log(np.exp(2)) # Log(e^2) = 2.0

# ### Crecimiento Compuesto
#
# Considera invertir $1,000 a una tasa anual del 5%:
#
# $$V_t = V_0 \times (1 + r)^t$$
#
# Donde:
# - $V_t$ = Valor en el tiempo $t$
# - $V_0$ = Valor inicial
# - $r$ = Tasa de crecimiento (0.05 = 5%)
# - $t$ = Número de períodos

# Año 1
print(f"Year 1: ${1000 * 1.05:.2f}")

# Año 2
print(f"Year 2: ${1000 * 1.05 * 1.05:.2f}")

# Año 3
print(f"Year 3: ${1000 * 1.05 * 1.05 * 1.05:.2f}")

# Usando un bucle
capital = 1000
for year in range(1, 21):
    capital *= 1.05
    if year % 5 == 0:
        print(f"Year {year}: ${capital:.2f}")

# Usando la fórmula directamente
1000 * 1.05 ** 20 # 1000 * (1.05^20) = 2653.297705144422

# ### ¿Cuánto Tiempo para Duplicar tu Inversión?
#
# Queremos encontrar $t$ tal que:
#
# $$2V_0 = V_0 \times (1 + r)^t$$
#
# Simplificando:
#
# $$2 = (1 + r)^t$$
#
# Tomando el logaritmo natural de ambos lados:
#
# $$\ln(2) = t \times \ln(1 + r)$$
#
# Por lo tanto:
#
# $$t = \frac{\ln(2)}{\ln(1 + r)}$$

r = 0.05  # 5% retorno anual
t = np.log(2) / np.log(1 + r)
print(f"Time to double at 5% annual return: {t:.2f} years")

# Verificar
1000 * 1.05 ** t

# **Regla del 72**: Una aproximación rápida es $t \approx 72 / (r \times 100)$. Al 5%, esto da $72/5 = 14.4$ años.
#
# ---
#
# ## Log Returns
#
# ### ¿Por Qué Retornos en Lugar de Precios?
#
# Los precios brutos no son directamente comparables entre diferentes activos o períodos de tiempo.
# Una ganancia de $100 significa cosas diferentes dependiendo del capital:

pnl = 100

# Ganancia de $100 sobre inversión de $50 = 200% de retorno
print(f"$100 on $50: {pnl / 50:.0%} return")

# Ganancia de $100 sobre inversión de $99 = ~101% de retorno
print(f"$100 on $99: {pnl / 99:.1%} return")

# ### Retornos Simples vs Log Returns
#
# **Retorno Simple**:
#
# $$R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$
#
# **Log Return** (Retorno Continuamente Compuesto):
#
# $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$
#
# ### Por Qué los Log Returns son Preferidos en Finanzas Cuantitativas
#
# 1. **Aditividad**: Los log returns se suman a través del tiempo: $r_{total} = r_1 + r_2 + ... + r_n$
# 2. **Simetría**: Un log return de +10% seguido de -10% retorna al valor original
# 3. **Propiedades Estadísticas**: Más probable que sigan una distribución normal
# 4. **Estabilidad Numérica**: Sin problemas de composición en períodos largos

# Ejemplo: El portafolio va $100 -> $120 -> $100
portfolio = [100, 120, 100]

# Retornos simples
simple_returns = [(120-100)/100, (100-120)/120]
print(f"Simple returns: {simple_returns}")
print(f"Sum of simple returns: {sum(simple_returns):.4f}")  # ¡No es cero!

# Log returns
log_returns = [np.log(120/100), np.log(100/120)]
print(f"Log returns: {log_returns}")
print(f"Sum of log returns: {sum(log_returns):.6f}")  # ¡Cero! (dentro de la precisión de punto flotante)

# **Punto Clave**: Los log returns son aditivos, lo que los hace ideales para:
# - Calcular retornos acumulados
# - Agregación de portafolios
# - Análisis estadístico
# - Entrenamiento de modelos
#
# ---
#
# ## Ejercicios Prácticos
#
# ### Ejercicio 1: Calcular el Log Return Promedio
#
# Dada una serie de log returns, calcula el promedio (media) del log return usando un bucle.

log_returns = [-0.1, 0.22, 0.15, 0.344, -0.2]
avg_log_return = 0.0

# TODO: Escribe un bucle para calcular el log return promedio
# Pista: suma todos los retornos, luego divide por la cantidad

# Tu bucle va aquí

# Verifica tu respuesta
expected = 0.0828
print(f"Your answer: {avg_log_return}")
print(f"Expected: {expected}")
print(f"Correct: {abs(avg_log_return - expected) < 0.0001}")

# ### Ejercicio 2: Calcular Log Returns Totales
#
# Dada una serie de valores de portafolio, calcula los log returns y verifica que la suma de log returns
# sea igual al log return total desde el inicio hasta el final.

portfolio = [100, 120, 100, 80, 155]
log_returns = []

# TODO: Calcular log returns entre valores consecutivos del portafolio
# Pista: log_return = np.log(P_t / P_{t-1})

# Tu bucle va aquí

# Verificar: valor inicial * exp(suma de log returns) debería ser igual al valor final
total_log_return = np.sum(log_returns) if log_returns else 0
final_value = 100 * np.exp(total_log_return)
print(f"Calculated final value: {final_value:.1f}")
print(f"Actual final value: 155.0")
print(f"Correct: {abs(final_value - 155.0) < 0.01}")

# ### Ejercicio 3: Calcular Log Returns Acumulados
#
# Calcula los log returns acumulados en cada paso temporal.

portfolio = [100, 120, 100, 80, 155]
cum_log_returns = []

# TODO: Calcular log returns acumulados
# Cada entrada debería ser la suma de todos los log returns hasta ese punto

# Tu bucle va aquí

# Verificar
expected = [
    np.log(120/100),
    np.log(120/100) + np.log(100/120),
    np.log(120/100) + np.log(100/120) + np.log(80/100),
    np.log(120/100) + np.log(100/120) + np.log(80/100) + np.log(155/80)
]
print(f"Your answer: {cum_log_returns}")
print(f"Expected: {expected}")

# ### Ejercicio de Desafío: Seguimiento de Portafolio
#
# Implementa una función que tome una lista de precios y retorne:
# 1. Log returns
# 2. Log returns acumulados
# 3. Valor del portafolio en ejecución (comenzando con $100)

def analyze_portfolio(prices):
    """
    Analyze a price series and return portfolio metrics.

    Parameters:
    -----------
    prices : list
        List of prices

    Returns:
    --------
    dict with keys:
        - 'log_returns': list of log returns
        - 'cum_log_returns': list of cumulative log returns
        - 'portfolio_value': list of portfolio values starting at $100
    """
    # TODO: Implementa esta función
    pass

# Prueba tu función
test_prices = [100, 105, 103, 110, 108]
result = analyze_portfolio(test_prices)
if result:
    print(f"Log Returns: {result['log_returns']}")
    print(f"Cumulative Returns: {result['cum_log_returns']}")
    print(f"Portfolio Value: {result['portfolio_value']}")

# ---
#
# ## Puntos Clave
#
# 1. **Los arrays son fundamentales**: Cada aspecto del trading cuantitativo depende de operaciones con arrays
#
# 2. **La complejidad computacional importa**: Usa `pop()` en lugar de `pop(0)` para rendimiento O(1)
#
# 3. **NumPy es esencial**: Las operaciones vectorizadas son órdenes de magnitud más rápidas que los bucles
#
# 4. **Los log returns son preferidos** porque son:
#    - Aditivos a través del tiempo
#    - Simétricos (ganancias y pérdidas son comparables)
#    - Más probable que sigan distribución normal
#
# 5. **Fórmulas clave**:
#    - Log return: $r_t = \ln(P_t / P_{t-1})$
#    - Retorno total: $r_{total} = \sum_{i=1}^{n} r_i$
#    - Valor final: $V_n = V_0 \times e^{r_{total}}$
#
# ---
#
# **Siguiente Módulo**: Vectorización - Construyendo herramientas eficientes de análisis de datos usando operaciones vectorizadas
