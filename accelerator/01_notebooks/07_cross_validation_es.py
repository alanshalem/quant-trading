#!/usr/bin/env python
# coding: utf-8

# # Módulo 07: Cross-Validation
#
# **Quant Trading Accelerator**
#
# ---

# ## Tabla de Contenidos
#
# 1. [Objetivos de Aprendizaje](#learning-objectives)
# 2. [¿Por Qué Cross-Validation?](#why-cross-validation)
# 3. [Time Series Split](#time-series-split)
# 4. [Expanding Window](#expanding-window)
# 5. [Rolling Window](#rolling-window)
# 6. [Comparando Métodos de CV](#comparing-cv-methods)
# 7. [Ejercicios Prácticos](#practical-exercises)
# 8. [Puntos Clave](#key-takeaways)

# ---
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este módulo, serás capaz de:
#
# - Entender por qué cross-validation es crucial para la evaluación de modelos
# - Implementar validación time series split
# - Implementar cross-validation con expanding window
# - Implementar cross-validation con rolling window
# - Comparar el rendimiento del modelo a través de diferentes métodos de CV
# - Elegir la estrategia de CV apropiada para tu sistema de trading

# Librerías principales
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning
import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibilidad
import random
import os

# ---
#
# ## ¿Por Qué Cross-Validation?
#
# ### El Problema con un Único Split Train/Test
#
# Un único split train/test puede ser:
# - **Sensible al punto de corte**: Diferentes splits pueden dar resultados muy distintos
# - **No representativo**: El período de test podría ser inusualmente fácil o difícil
# - **Sobreajustado a un período**: El modelo podría funcionar solo para ese rango temporal específico
#
# ### Beneficios de Cross-Validation
#
# 1. **Estimaciones más robustas**: Rendimiento promedio a través de múltiples períodos
# 2. **Estimación de varianza**: Ver cómo varía el rendimiento
# 3. **Mejor generalización**: Prueba el modelo en diferentes condiciones de mercado
# 4. **Detectar overfitting**: Rendimiento inconsistente señala problemas

# ---
#
# ## Métodos de Cross-Validation para Series Temporales
#
# **Importante**: ¡El k-fold CV estándar viola el orden temporal! Necesitamos métodos especializados:
#
# ### 1. Time Series Split
# ```
# Tiempo: t0 ---- t1 ---- t2 ---- t3 ---- t4 ---- t5 ---- t6 ---- t7
# Train:  [===============================]
# Test:                                   [=========================]
# ```
#
# ### 2. Expanding Window
# ```
# Fold 1: Train [###]     Test [--]
# Fold 2: Train [######]  Test    [--]
# Fold 3: Train [#########] Test      [--]
# ```
#
# ### 3. Rolling Window
# ```
# Fold 1: Train [###]     Test [--]
# Fold 2:       Train [###]    Test [--]
# Fold 3:            Train [###]    Test [--]
# ```

# ---
#
# ## Cargar y Preparar Datos

url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
btcusdt = pd.read_csv(url, parse_dates=["open_time"], index_col='open_time')

# Ingeniería de features
btcusdt['close_log_return'] = np.log(btcusdt['close'] / btcusdt['close'].shift())
btcusdt['close_log_return_lag_1'] = btcusdt['close_log_return'].shift(1)
btcusdt['close_log_return_lag_2'] = btcusdt['close_log_return'].shift(2)
btcusdt['close_log_return_lag_3'] = btcusdt['close_log_return'].shift(3)
btcusdt = btcusdt.dropna()

print(f"Total samples: {len(btcusdt)}")
btcusdt.head()

# ---
#
# ## Funciones Auxiliares

def train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, no_epochs, verbose=True):
    """
    Train a PyTorch model with reproducible settings.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train
    criterion : loss function
        Loss function (e.g., nn.HuberLoss)
    optimizer : optimizer
        PyTorch optimizer
    X_train, y_train : tensors
        Training data
    X_test, y_test : tensors
        Test data (unused during training, for reference)
    no_epochs : int
        Number of training epochs
    verbose : bool
        Print training progress

    Returns
    -------
    model : trained model
    """
    # Reproducibilidad
    SEED = 99
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Bucle de entrenamiento
    for epoch in range(no_epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    if verbose:
        print(f"Trained weights: {model.weight.data}")
        print(f"Trained bias: {model.bias.data}")

    return model

def test_model_predictions(model, X_test):
    """Get model predictions on test data."""
    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)
    return y_hat.squeeze(1)

def eval_profitability(model, df_test, X_test, target='close_log_return'):
    """
    Evaluate model profitability on test set.

    Returns expected trade log return.
    """
    y_hat = test_model_predictions(model, X_test).numpy()
    df_test = df_test.copy()
    df_test['y_hat'] = y_hat
    df_test['dir_signal'] = np.sign(y_hat)
    df_test['trade_log_return'] = df_test['dir_signal'] * df_test[target]
    df_test['is_won'] = df_test['trade_log_return'] > 0

    return df_test['trade_log_return'].mean()

def eval_model_profitability(df_train, df_test, features, target):
    """
    Train model and evaluate profitability.

    Returns expected trade log return on test set.
    """
    no_features = len(features)

    # Crear modelo
    model = nn.Linear(no_features, 1)
    criterion = nn.HuberLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Convertir a tensores
    X_train = torch.tensor(df_train[features].values, dtype=torch.float32)
    X_test = torch.tensor(df_test[features].values, dtype=torch.float32)
    y_train = torch.tensor(df_train[target].values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(df_test[target].values, dtype=torch.float32).unsqueeze(1)

    # Entrenar
    train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test,
                no_epochs=5000, verbose=False)

    # Evaluar
    return eval_profitability(model, df_test, X_test, target)

# ---
#
# ## Time Series Split
#
# Split cronológico simple train/test a diferentes ratios.

def timesplit(df, train_size=0.75):
    """Split data chronologically."""
    i = int(len(df) * train_size)
    return df[:i].copy(), df[i:].copy()

# Probar diferentes ratios de split
features = ['close_log_return_lag_1', 'close_log_return_lag_2', 'close_log_return_lag_3']
target = 'close_log_return'

print("Time Series Split a Diferentes Tamaños de Train:")
print("-" * 50)

for train_ratio in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    btcusdt_train, btcusdt_test = timesplit(btcusdt, train_size=train_ratio)
    ev = eval_model_profitability(btcusdt_train, btcusdt_test, features, target)
    print(f"Train {train_ratio:.0%} | Test {1-train_ratio:.0%} | E[Trade Return]: {ev:.6f}")

# ### Agregar a Través de Múltiples Splits

train_sizes = []
evs = []

for train_split in [0.4 + 0.1 * i for i in range(6)]:
    btcusdt_train, btcusdt_test = timesplit(btcusdt, train_size=train_split)
    ev = eval_model_profitability(btcusdt_train, btcusdt_test, features, target)
    train_sizes.append(train_split)
    evs.append(ev)

cv_results = pd.DataFrame({'train_size': train_sizes, 'ev': evs})
cv_results

print(f"Mean Expected Value: {cv_results['ev'].mean():.6f}")
print(f"Std of Expected Value: {cv_results['ev'].std():.6f}")

# ---
#
# ## Cross-Validation con Rolling Window
#
# Tamaño de ventana fijo que se desplaza hacia adelante en el tiempo.
#
# ```
# Tiempo:    [=========================================]
#             t1  t2  t3  t4  t5  t6  t7  t8  t9  t10
#
# Fold 1:   [####]  [--]
# Fold 2:       [####]  [--]
# Fold 3:           [####]  [--]
# Fold 4:               [####]  [--]
# ```

# Calcular tamaño de ventana (aproximadamente 1 mes de datos por hora)
hours_in_month = 24 * 30
print(f"Hours in a month: {hours_in_month}")
print(f"Total rows: {len(btcusdt)}")
print(f"Number of possible monthly windows: {len(btcusdt) / hours_in_month:.1f}")

def eval_rolling_window_cv(df, features, target, window_size, no_iterations):
    """
    Evaluate model using rolling window cross-validation.

    Parameters
    ----------
    df : DataFrame
        Full dataset
    features : list
        Feature column names
    target : str
        Target column name
    window_size : int
        Size of each window (train and test)
    no_iterations : int
        Number of rolling iterations

    Returns
    -------
    DataFrame with results for each fold
    """
    window_no = []
    ev = []

    for i in range(no_iterations):
        # Calcular índices
        train_start = window_size * i
        train_end = window_size * (i + 1)
        test_start = train_end
        test_end = test_start + window_size

        # Verificar límites
        if test_end > len(df):
            print(f"Warning: Fold {i} exceeds data bounds, stopping.")
            break

        # Dividir datos
        df_train = df.iloc[train_start:train_end].copy()
        df_test = df.iloc[test_start:test_end].copy()

        # Evaluar
        window_no.append(i)
        ev.append(eval_model_profitability(df_train, df_test, features, target))

    return pd.DataFrame({'window_no': window_no, 'ev': ev})

# Ejecutar rolling window CV
window_size = 724  # ~1 mes de datos por hora
rw_results = eval_rolling_window_cv(btcusdt, features, target, window_size, 6)
rw_results

print(f"Rolling Window CV - Mean E[Return]: {rw_results['ev'].mean():.6f}")
print(f"Rolling Window CV - Std E[Return]: {rw_results['ev'].std():.6f}")

# Visualizar resultados
plt.figure(figsize=(12, 6))
plt.bar(rw_results['window_no'], rw_results['ev'])
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=rw_results['ev'].mean(), color='g', linestyle='--', alpha=0.5, label='Mean')
plt.xlabel('Window Number')
plt.ylabel('Expected Trade Return')
plt.title('Rolling Window Cross-Validation Results')
plt.legend()
plt.show()

# ---
#
# ## Cross-Validation con Expanding Window
#
# La ventana de entrenamiento se expande con el tiempo mientras la ventana de test permanece fija.
#
# ```
# Tiempo:    [=========================================]
#             t1  t2  t3  t4  t5  t6  t7  t8  t9  t10
#
# Fold 1:   [####]  [--]
# Fold 2:   [########]  [--]
# Fold 3:   [############]  [--]
# Fold 4:   [################]  [--]
# ```

def eval_expanding_window_cv(df, features, target, window_size, no_iterations):
    """
    Evaluate model using expanding window cross-validation.

    Parameters
    ----------
    df : DataFrame
        Full dataset
    features : list
        Feature column names
    target : str
        Target column name
    window_size : int
        Size of each test window and initial train window
    no_iterations : int
        Number of expanding iterations

    Returns
    -------
    DataFrame with results for each fold
    """
    iteration_no = []
    ev = []

    for i in range(no_iterations):
        # La ventana de train se expande, la de test tiene tamaño fijo
        train_start = 0
        train_end = window_size + i * window_size
        test_start = train_end
        test_end = test_start + window_size

        # Verificar límites
        if test_end > len(df):
            print(f"Warning: Fold {i} exceeds data bounds, stopping.")
            break

        # Dividir datos
        df_train = df.iloc[train_start:train_end].copy()
        df_test = df.iloc[test_start:test_end].copy()

        # Evaluar
        iteration_no.append(i + 1)
        ev.append(eval_model_profitability(df_train, df_test, features, target))

    return pd.DataFrame({'iteration_no': iteration_no, 'ev': ev})

# Mostrar índices de expanding window
window_size = 724
print("Expanding Window Indices:")
print("-" * 50)
for i in range(6):
    train_start = 0
    train_end = window_size + i * window_size
    test_start = train_end
    test_end = test_start + window_size
    print(f"Fold {i+1}: Train[{train_start}:{train_end}] ({train_end} samples) | Test[{test_start}:{test_end}]")

# Ejecutar expanding window CV
ew_results = eval_expanding_window_cv(btcusdt, features, target, window_size, 6)
ew_results

print(f"Expanding Window CV - Mean E[Return]: {ew_results['ev'].mean():.6f}")
print(f"Expanding Window CV - Std E[Return]: {ew_results['ev'].std():.6f}")

# Visualizar resultados
plt.figure(figsize=(12, 6))
plt.bar(ew_results['iteration_no'], ew_results['ev'])
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=ew_results['ev'].mean(), color='g', linestyle='--', alpha=0.5, label='Mean')
plt.xlabel('Iteration Number')
plt.ylabel('Expected Trade Return')
plt.title('Expanding Window Cross-Validation Results')
plt.legend()
plt.show()

# ---
#
# ## Comparando Métodos de CV

# Resumen de comparación
comparison = pd.DataFrame({
    'Method': ['Time Series Split', 'Rolling Window', 'Expanding Window'],
    'Mean E[Return]': [cv_results['ev'].mean(), rw_results['ev'].mean(), ew_results['ev'].mean()],
    'Std E[Return]': [cv_results['ev'].std(), rw_results['ev'].std(), ew_results['ev'].std()]
})
comparison

# Comparación visual
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(range(len(cv_results)), cv_results['ev'])
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_title(f"Time Series Split\nMean: {cv_results['ev'].mean():.6f}")
axes[0].set_xlabel('Split')
axes[0].set_ylabel('E[Trade Return]')

axes[1].bar(range(len(rw_results)), rw_results['ev'])
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_title(f"Rolling Window\nMean: {rw_results['ev'].mean():.6f}")
axes[1].set_xlabel('Window')

axes[2].bar(range(len(ew_results)), ew_results['ev'])
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_title(f"Expanding Window\nMean: {ew_results['ev'].mean():.6f}")
axes[2].set_xlabel('Iteration')

plt.tight_layout()
plt.show()

# ---
#
# ## Eligiendo una Estrategia de CV
#
# | Método | Ventajas | Desventajas | Mejor Para |
# |--------|----------|-------------|------------|
# | **Time Series Split** | Simple, rápido | Una sola estimación | Verificaciones rápidas |
# | **Rolling Window** | Tamaño de train consistente, se adapta a cambios de régimen | Descarta datos antiguos | Mercados no estacionarios |
# | **Expanding Window** | Usa todos los datos disponibles, estable | El tamaño de train varía | Mercados estables |

# ---
#
# ## Ejercicios Prácticos
#
# ### Ejercicio 1: Optimizar Tamaño de Rolling Window
#
# Prueba diferentes tamaños de ventana para encontrar el valor óptimo.

# TODO: Probar tamaños de ventana de 500 a 1000 horas
# Encontrar el tamaño de ventana con mejor E[Return] promedio

# ### Ejercicio 2: Optimizar Tamaño Inicial de Expanding Window
#
# Experimenta con diferentes tamaños de ventana inicial.

# TODO: Probar diferentes tamaños iniciales de train
# Comparar resultados

# ### Ejercicio 3: Walk-Forward Optimization
#
# Implementa walk-forward optimization donde los hiperparámetros del modelo
# se ajustan en los datos de entrenamiento de cada fold.

# TODO: Para cada fold:

# 1. Dividir train en train/validación
# 2. Ajustar hiperparámetros en validación
# 3. Reentrenar en train completo con mejores hiperparámetros
# 4. Evaluar en test

# ---
#
# ## Puntos Clave
#
# 1. **Un único split train/test no es confiable**
#    - Los resultados dependen fuertemente de dónde divides
#    - Usa múltiples evaluaciones para estimaciones robustas
#
# 2. **Las series temporales requieren métodos especiales de CV**
#    - Nunca mezcles datos de series temporales
#    - Preserva el orden temporal en todas las divisiones
#
# 3. **Rolling Window CV**:
#    - Ventana de train fija que se desliza a través del tiempo
#    - Bueno para mercados no estacionarios
#    - Se adapta a cambios de régimen
#
# 4. **Expanding Window CV**:
#    - La ventana de train crece con el tiempo
#    - Usa todos los datos históricos
#    - Mejor para relaciones estables
#
# 5. **Métricas de evaluación**:
#    - Mira la **media** y **std** de los resultados
#    - Alta varianza sugiere que el modelo es inestable
#    - E[Return] positivo consistente indica edge robusto
#
# ---
#
# **Siguiente Módulo**: Strategy Logic - Construyendo una estrategia de trading completa
