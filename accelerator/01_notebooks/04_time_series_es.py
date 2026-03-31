#!/usr/bin/env python
# coding: utf-8

# # Módulo 04: Análisis de Series Temporales
#
# **Quant Trading Accelerator**
#
# ---

# ## Tabla de Contenidos
#
# 1. [Objetivos de Aprendizaje](#learning-objectives)
# 2. [Fundamentos Estadísticos](#statistical-foundations)
# 3. [Tendencia Central](#central-tendency)
# 4. [Dispersión y Volatilidad](#spread-and-volatility)
# 5. [Retornos Ajustados por Riesgo](#risk-adjusted-returns)
# 6. [Correlación](#correlation)
# 7. [Fundamentos de Series Temporales](#time-series-fundamentals)
# 8. [Diferenciación y Estacionariedad](#differencing-and-stationarity)
# 9. [Modelos Autorregresivos](#autoregressive-models)
# 10. [Puntos Clave](#key-takeaways)

# ---
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este módulo, serás capaz de:
#
# - Aplicar conceptos estadísticos clave a series temporales financieras
# - Entender y calcular retornos ajustados por riesgo (Sharpe Ratio)
# - Identificar e interpretar patrones de correlación
# - Trabajar con datos reales OHLC de criptomonedas
# - Entender la estacionariedad y por qué la diferenciación importa
# - Crear features con lag para modelos autorregresivos
# - Entender las dinámicas de mean reversion y momentum

# ---
#
# ## Fundamentos Estadísticos
#
# El análisis de series temporales se construye sobre conceptos estadísticos fundamentales.
# Revisemos las medidas clave que usaremos a lo largo de nuestros modelos de trading.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---
#
# ## Tendencia Central
#
# ### Media (Promedio)
#
# La **media aritmética** es la suma de valores dividida por la cantidad:
#
# $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
#
# **En trading**: La media del PnL nos dice nuestro retorno esperado por operación.

trade_pnl = [10.0, 11.0, 9.0, -400.0, 2.0, 10.0, 11.0, 11.0, 8, 10.0, 10.2]
mean_pnl = np.mean(trade_pnl)
print(f"Mean P&L: ${mean_pnl:.2f}")

# **Problema**: La media está fuertemente influenciada por valores atípicos (la pérdida de -400).
#
# ### Mediana
#
# La **mediana** es el valor central cuando los datos están ordenados. Es robusta frente a valores atípicos:
#
# $$\text{Mediana} = x_{(n+1)/2} \text{ para } n \text{ impar}$$

median_pnl = np.median(trade_pnl)
print(f"Median P&L: ${median_pnl:.2f}")

# **Punto Clave**: La mediana ($10.0) es mucho más representativa de las operaciones típicas
# que la media (-$28.00), que está sesgada por una gran pérdida.
#
# ---
#
# ## Dispersión y Volatilidad
#
# ### Desviación Estándar
#
# La **desviación estándar** mide la distancia típica desde la media:
#
# $$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$
#
# **En trading**: La desviación estándar mide la **volatilidad** - el riesgo de los retornos.

# Portafolio A: Ganancias pequeñas consistentes
portfolio_a = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
print(f"Portfolio A - Total P&L: ${np.sum(portfolio_a):.2f}")
print(f"Portfolio A - Std Dev: ${np.std(portfolio_a):.4f}")

# Portafolio B: Grandes oscilaciones, mismo total
portfolio_b = [-10.0, 10.0, -10.0, 15.0, -10.0, 16.0, -18.0, 13.0, -10.0, 10.0]
print(f"Portfolio B - Total P&L: ${np.sum(portfolio_b):.2f}")
print(f"Portfolio B - Std Dev: ${np.std(portfolio_b):.4f}")

# **Punto Clave**: Ambos portafolios tienen un PnL total similar (~$6), ¡pero el Portafolio B
# es mucho más riesgoso con 20x más volatilidad!
#
# ---
#
# ## Retornos Ajustados por Riesgo
#
# ### Comparando Portafolios
#
# Los retornos brutos no cuentan toda la historia. Necesitamos métricas **ajustadas por riesgo**.

# Crear tabla de comparación
benchmark = pd.DataFrame([
    ['A', np.sum(portfolio_a), np.mean(portfolio_a), np.std(portfolio_a)],
    ['B', np.sum(portfolio_b), np.mean(portfolio_b), np.std(portfolio_b)]
], columns=['portfolio', 'total_pnl', 'mean_pnl', 'std_pnl'])

benchmark

# ### Sharpe Ratio
#
# El **Sharpe Ratio** mide el retorno por unidad de riesgo:
#
# $$\text{Sharpe Ratio} = \frac{\mathbb{E}[R] - R_f}{\sigma_R}$$
#
# Donde:
# - $\mathbb{E}[R]$ = Retorno esperado (media)
# - $R_f$ = Tasa libre de riesgo (frecuentemente 0 por simplicidad)
# - $\sigma_R$ = Desviación estándar de los retornos

benchmark['sharpe'] = benchmark['mean_pnl'] / benchmark['std_pnl']
benchmark

# **Interpretación**:
# - Portafolio A: Sharpe = 1.03 (excelentes retornos ajustados por riesgo)
# - Portafolio B: Sharpe = 0.05 (pobres retornos ajustados por riesgo)
#
# **¡Aunque ambos tienen PnL total similar, el Portafolio A es vastamente superior!**
#
# ---
#
# ## Correlación
#
# La **correlación** mide la relación lineal entre dos variables:
#
# $$\rho_{xy} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$
#
# La correlación va de -1 a +1:
# - **+1**: Correlación positiva perfecta
# - **0**: Sin relación lineal
# - **-1**: Correlación negativa perfecta
#
# ### Correlación Positiva

x = [1, 2, 3, 4]
y = [1, 3, 2, 5]

corr = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {corr:.4f} (Strong positive)")

# Correlación positiva perfecta
x = [1, 2, 3, 4]
y = [2, 3, 4, 5]

corr = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {corr:.4f} (Perfect positive)")

# ### Correlación Negativa

x = [1, 2, 3, 4]
y = [-1, -3, -2, -6]

corr = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {corr:.4f} (Strong negative)")

# Correlación negativa perfecta
x = [1, 2, 3, 4]
y = [-1, -2, -3, -4]

corr = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {corr:.4f} (Perfect negative)")

# **En trading**: Buscamos correlaciones entre retornos con lag y retornos futuros
# para encontrar patrones predecibles.
#
# ---
#
# ## Fundamentos de Series Temporales
#
# ### Cargando Datos Reales de Mercado
#
# Trabajemos con datos reales de futuros perpetuos BTCUSDT.

# Cargar datos OHLC por hora
url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')

print(f"Data shape: {btcusdt.shape}")
print(f"Date range: {btcusdt.index.min()} to {btcusdt.index.max()}")
btcusdt.head()

# ### Visualizando la Serie de Precios

btcusdt['close'].plot(figsize=(15, 6), title='BTC/USDT Close Price')
plt.ylabel('Price (USDT)')
plt.show()

# ### Distribución de Precios
#
# Los precios brutos **no siguen una distribución normal** - tienen una cola larga a la derecha:

btcusdt['close'].hist(bins=200, figsize=(15, 6))
plt.title("Distribution of Close Prices")
plt.xlabel("Close Price (USDT)")
plt.ylabel("Frequency")
plt.show()

print(f"Mean price: ${btcusdt['close'].mean():,.2f}")
print(f"Std dev: ${btcusdt['close'].std():,.2f}")

# ---
#
# ## Diferenciación y Estacionariedad
#
# ### El Problema con los Precios Brutos
#
# Los precios brutos son **no estacionarios** - sus propiedades estadísticas cambian con el tiempo.
# Esto los hace difíciles de modelar y predecir.
#
# ### Diferenciación
#
# La **diferenciación** convierte precios en cambios de precio:
#
# $$\Delta P_t = P_t - P_{t-1}$$

btcusdt['close_delta'] = btcusdt['close'] - btcusdt['close'].shift()
btcusdt[['close', 'close_delta']].head(10)

# ### Distribución de Cambios de Precio
#
# Los datos diferenciados están mucho más cerca de una distribución normal:

btcusdt['close_delta'].hist(bins=80, figsize=(15, 6))
plt.title("Distribution of Close Price Delta (More Normal!)")
plt.xlabel("Price Change (USDT)")
plt.ylabel("Frequency")
plt.show()

print(f"Mean price change: ${btcusdt['close_delta'].mean():.2f}")
print(f"Std dev: ${btcusdt['close_delta'].std():.2f}")

# ### Verificación de Estacionariedad: Estadísticas Anuales
#
# La verdadera estacionariedad requiere que las propiedades estadísticas permanezcan constantes en el tiempo.
# Verifiquemos si nuestra serie diferenciada es realmente estacionaria:

yearly_stats = btcusdt['close_delta'].resample('YE').agg(['mean', 'std'])
yearly_stats

# **Advertencia**: ¡La desviación estándar varía significativamente por año!
# Esto es **heteroscedasticidad** (volatilidad cambiante), común en datos financieros.
# La diferenciación sola no garantiza estacionariedad.
#
# ---
#
# ## Creando Features con Lag
#
# ### Operaciones de Lag
#
# Los **lags** desplazan datos un número especificado de períodos, creando features
# para modelos autorregresivos:
#
# $$x_{t-k} = \text{lag}(x_t, k)$$

btcusdt['close_delta_lag_1'] = btcusdt['close_delta'].shift(1)
btcusdt['close_delta_lag_2'] = btcusdt['close_delta'].shift(2)
btcusdt['close_delta_lag_3'] = btcusdt['close_delta'].shift(3)
btcusdt['close_delta_lag_4'] = btcusdt['close_delta'].shift(4)

btcusdt[['close_delta', 'close_delta_lag_1', 'close_delta_lag_2',
         'close_delta_lag_3', 'close_delta_lag_4']].head(10)

# Eliminar filas NaN para análisis
btcusdt[['close_delta', 'close_delta_lag_1', 'close_delta_lag_2',
         'close_delta_lag_3', 'close_delta_lag_4']].dropna().head(10)

# ---
#
# ## Correlación Serial
#
# La **correlación serial** (autocorrelación) mide cuán correlacionada está una serie temporal
# con sus propios valores pasados. ¡Esto es crucial para encontrar predictibilidad!
#
# $$\rho_k = \text{Corr}(x_t, x_{t-k})$$

# Matriz de correlación de deltas con sus lags
corr_matrix = btcusdt[['close_delta', 'close_delta_lag_1', 'close_delta_lag_2',
                       'close_delta_lag_3', 'close_delta_lag_4']].dropna().corr()
corr_matrix

# **Interpretación**:
# - Mira la primera fila (correlaciones de close_delta)
# - Pequeñas correlaciones negativas con lags sugieren débil **mean reversion**
# - Si encontráramos correlaciones positivas, sugeriría **momentum**
#
# ---
#
# ## Modelos Autorregresivos
#
# ### AR(1): Modelo Autorregresivo de Primer Orden
#
# El **modelo AR(1)** predice el siguiente valor basándose en el valor anterior:
#
# $$y_t = w \cdot y_{t-1} + b + \epsilon_t$$
#
# Donde:
# - $w$ = Peso (coeficiente)
# - $b$ = Sesgo (intercepto)
# - $\epsilon_t$ = Término de error aleatorio

# Crear serie temporal simple
ts = pd.DataFrame({'log_return': [-0.1, 0.2, -0.2, 0.1, -0.3, 0.3]})

# Agregar índice de fechas
n = len(ts)
dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq='D')
ts.index = dates

ts

# Crear feature de lag
ts['log_return_lag_1'] = ts['log_return'].shift()
ts

# ### Definición del Modelo AR(1)

# Modelo AR(1): y = w * y_lag_1 + b
y = ts['log_return']
y_lag_1 = ts['log_return_lag_1']

# Parámetros iniciales (sin entrenar)
weight, bias = 0.0, 0.0

# Predicción (sería 0 con estos parámetros)
y_hat = weight * y_lag_1 + bias

# ---
#
# ## Dinámicas Fundamentales de Trading
#
# El modelo AR(1) puede capturar dos comportamientos fundamentales del mercado:
#
# 1. **Mean Reversion**: Los precios tienden a retornar a su promedio ($w < 0$)
# 2. **Momentum**: Los precios continúan en su dirección actual ($w > 0$)
#
# ### Mean Reversion
#
# En mercados con mean reversion, un retorno positivo es seguido por un retorno negativo,
# y viceversa. El peso $w$ es **negativo**.

weight = -0.5  # Peso negativo = mean reversion
bias = 0.00001

ts['y_hat_reversion'] = weight * ts['log_return_lag_1'] + bias
ts[['log_return', 'log_return_lag_1', 'y_hat_reversion']]

# **Interpretación**: Cuando el retorno de ayer fue positivo (+0.2), nuestro modelo
# predice un retorno negativo hoy (-0.1). Este es el comportamiento de mean reversion.
#
# ### Momentum
#
# En mercados con tendencia, un retorno positivo es seguido por otro retorno positivo.
# El peso $w$ es **positivo**.

# Crear serie con tendencia
ts_momentum = pd.DataFrame({
    'log_return': [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]
})

n = len(ts_momentum)
dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq='D')
ts_momentum.index = dates
ts_momentum['log_return_lag_1'] = ts_momentum['log_return'].shift()
ts_momentum

weight = 0.5  # Peso positivo = momentum
bias = 0.0001

ts_momentum['y_hat_momentum'] = weight * ts_momentum['log_return_lag_1'] + bias
ts_momentum[['log_return', 'log_return_lag_1', 'y_hat_momentum']]

# **Interpretación**: Cuando el retorno de ayer fue positivo (+0.1), nuestro modelo
# predice otro retorno positivo (+0.05). Este es el comportamiento de momentum.
#
# ---
#
# ## Resumen: Mean Reversion vs Momentum
#
# | Comportamiento | Peso AR(1) | Patrón | Estrategia |
# |----------------|-----------|--------|------------|
# | Mean Reversion | $w < 0$ | Oscilación +/- | Ir contra el movimiento |
# | Random Walk | $w = 0$ | Sin patrón | No se puede predecir |
# | Momentum | $w > 0$ | Tendencias continúan | Seguir tendencias |

# ---
#
# ## Ejercicios Prácticos
#
# ### Ejercicio 1: Calcular Correlación Serial
#
# Calcula la correlación entre log returns y sus valores lag-1.

# TODO: Agregar log returns a btcusdt
# btcusdt['log_return'] = ...

# TODO: Calcular correlación entre log_return y log_return_lag_1

# ### Ejercicio 2: Identificar Régimen de Mercado
#
# Basándote en la correlación que encontraste, ¿BTC muestra mean reversion o momentum?

# TODO: Analizar e interpretar la correlación

# ### Ejercicio 3: Comparar Diferentes Períodos de Lag
#
# Crea features lag-1, lag-2 y lag-3 para log returns.
# ¿Cuál lag tiene la correlación más fuerte?

# TODO: Crear múltiples lags y comparar correlaciones

# ---
#
# ## Puntos Clave
#
# 1. **Tendencia central**: La media es sensible a valores atípicos; la mediana es robusta
#
# 2. **Retornos ajustados por riesgo**: Sharpe Ratio = retorno / riesgo
#    - Alto Sharpe = buen rendimiento ajustado por riesgo
#
# 3. **Correlación**: Mide la relación lineal (-1 a +1)
#    - La correlación serial revela patrones predecibles
#
# 4. **Estacionariedad**: Los datos financieros son frecuentemente no estacionarios
#    - La diferenciación ayuda pero no garantiza estacionariedad
#    - Los log returns son preferidos sobre niveles de precio
#
# 5. **Modelo AR(1)**: $y_t = w \cdot y_{t-1} + b$
#    - $w < 0$: Mean reversion
#    - $w > 0$: Momentum
#
# 6. **Fórmulas clave**:
#    - Sharpe Ratio: $SR = \frac{\bar{r}}{\sigma_r}$
#    - Correlación: $\rho_{xy} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$
#    - AR(1): $y_t = w \cdot y_{t-1} + b$
#
# ---
#
# **Siguiente Módulo**: Statistical Edge - Álgebra matricial y construcción de modelos ML para trading
