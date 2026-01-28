# Guia Completa: Construccion de una Estrategia de Trading Cuantitativo

## Tabla de Contenidos

1. [Introduccion](#1-introduccion)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Parte 1: Investigacion y Modelo](#3-parte-1-investigacion-y-modelo)
4. [Parte 2: Desarrollo de Estrategia](#4-parte-2-desarrollo-de-estrategia)
5. [Parte 3: Implementacion](#5-parte-3-implementacion)
6. [Metricas de Evaluacion](#6-metricas-de-evaluacion)
7. [Estrategias Adicionales](#7-estrategias-adicionales)
8. [Mejores Practicas](#8-mejores-practicas)

---

## 1. Introduccion

### 1.1 El Pipeline de Trading Cuantitativo

```python
y_hat = model(x)        # Prediccion del modelo
orders = strategy(y_hat) # Estrategia genera ordenes
execute(orders)          # Ejecucion en el mercado
```

### 1.2 Componentes Principales

| Componente | Descripcion | Archivo |
|------------|-------------|---------|
| Research | Funciones de investigacion y backtesting | `research.py` |
| Models | Definicion de modelos ML | `models.py` |
| Binance | API para datos de mercado | `binance.py` |
| Strategy | Logica de trading | `video2.py`, `video3.py` |

---

## 2. Arquitectura del Sistema

### 2.1 Estructura de Archivos

```
build-a-quant-trading-strategy/
├── research.py          # Utilidades de investigacion
├── models.py            # Modelos de ML
├── binance.py           # API de datos
├── video1.py            # Construccion del modelo
├── video2.py            # Desarrollo de estrategia
├── video3.py            # Implementacion en tiempo real
├── cache/               # Datos descargados
└── model_weights.pth    # Pesos del modelo entrenado
```

### 2.2 Dependencias Principales

```python
import polars as pl          # DataFrames de alto rendimiento
import torch                 # Framework de deep learning
import torch.nn as nn        # Capas neuronales
import numpy as np           # Operaciones numericas
import altair as alt         # Visualizacion interactiva
```

---

## 3. Parte 1: Investigacion y Modelo

### 3.1 Configuracion Inicial

```python
import research

# Semilla para reproducibilidad
research.set_seed(42)

# Parametros de trading
symbol = 'BTCUSDT'
time_interval = '12h'
forecast_horizon = 1
max_lags = 3
```

### 3.2 Carga de Datos

```python
from datetime import datetime

# Definir rango de fechas
start_date = datetime(2024, 10, 29)
end_date = datetime(2025, 10, 9)

# Descargar datos
import binance
binance.download_date_range(symbol, start_date, end_date)

# Cargar serie temporal OHLC
ohlc = research.load_ohlc_timeseries_range(
    symbol, time_interval, start_date, end_date
)
```

### 3.3 Feature Engineering

#### 3.3.1 Log Returns

Los log returns son preferidos porque:
- **Aditividad temporal**: `log(P2/P0) = log(P2/P1) + log(P1/P0)`
- **Simetria**: Movimientos arriba/abajo son simetricos
- **Propiedades estadisticas**: Mejor comportamiento para modelado

```python
# Calcular log returns
ohlc = ohlc.with_columns(
    (pl.col('close') / pl.col('close').shift(forecast_horizon))
    .log()
    .alias('close_log_return')
)
```

#### 3.3.2 Features Autoregresivos (Lags)

```python
target = 'close_log_return'

# Agregar lags automaticamente
ohlc = research.add_lags(ohlc, target, max_lags, forecast_horizon)

# Resultado: close_log_return_lag_1, _lag_2, _lag_3
```

### 3.4 Construccion del Modelo

#### 3.4.1 Modelo Lineal (Baseline)

```python
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

#### 3.4.2 Por que Modelos Lineales?

| Ventaja | Descripcion |
|---------|-------------|
| Interpretabilidad | Pesos tienen significado directo |
| Robustez | Menos sensibles al ruido |
| Generalizacion | Menor riesgo de overfitting |
| Velocidad | Inferencia rapida |

### 3.5 Entrenamiento

```python
# Split temporal (NO aleatorio)
test_size = 0.25
X_train, X_test, y_train, y_test = research.timeseries_train_test_split(
    ohlc, features, target, test_size
)

# Entrenar modelo
model = LinearModel(len(features))
y_hat = research.batch_train_reg(
    model, X_train, X_test, y_train, y_test,
    no_epochs=6000,
    optimizer_type='lbfgs'
)
```

### 3.6 Evaluacion del Modelo

```python
# Metricas de performance
perf = research.eval_model_performance(
    y_test, y_hat, features, target, annualized_rate
)

# Metricas clave:
# - win_rate: Porcentaje de trades correctos
# - sharpe: Sharpe ratio anualizado
# - max_drawdown: Maxima caida del equity
# - ev: Expected value por trade
```

### 3.7 Guardar Modelo

```python
torch.save(model.state_dict(), 'model_weights.pth')
```

---

## 4. Parte 2: Desarrollo de Estrategia

### 4.1 Tipos de Estrategias

| Tipo | Descripcion | Fees |
|------|-------------|------|
| **Maker** | Limit orders, provee liquidez | Menores (rebates) |
| **Taker** | Market orders, consume liquidez | Mayores |

### 4.2 Decisiones Clave de Estrategia

#### Decision 1: Entry/Exit Signal

```python
# Signal basado en prediccion del modelo
trades = trades.with_columns(
    pl.col('y_hat').sign().alias('dir_signal')
    # +1 = Long, -1 = Short
)
```

#### Decision 2: Trade Sizing

**Constant Sizing:**
```python
trade_value = capital * ratio  # Fijo por trade
```

**Compounding Sizing:**
```python
# El tamano crece/decrece con el equity
entry_value = capital * exp(cumulative_log_return)
```

#### Decision 3: Leverage

```python
leverage = 4  # 4x multiplicador
effective_capital = capital * leverage
```

**Reglas de Leverage:**
- Low Sharpe (< 0.5): Sin leverage
- Medium Sharpe (~1.0): 2-3x
- High Sharpe (> 2.0): Escalable

### 4.3 Transaction Fees

```python
taker_fee = 0.0003  # 0.03%
maker_fee = 0.0001  # 0.01%

# Fee por roundtrip (entry + exit)
roundtrip_fee_log = np.log(1 - 2 * taker_fee)
```

### 4.4 Calculo de PnL

```python
# Trade log return
trades = trades.with_columns(
    (pl.col('close_log_return') * pl.col('dir_signal'))
    .alias('trade_log_return')
)

# Equity curve
trades = trades.with_columns(
    pl.col('trade_log_return').cum_sum().alias('equity_curve')
)

# Net PnL (despues de fees)
trades = trades.with_columns(
    (pl.col('trade_gross_pnl') - pl.col('taker_fee'))
    .alias('trade_net_pnl')
)
```

### 4.5 Liquidation Risk

```python
maintenance_margin = 0.005  # 0.5%

def long_liquidation_price(entry_price, leverage, mmr):
    return (entry_price * leverage) / (leverage + 1 - mmr * leverage)

def short_liquidation_price(entry_price, leverage, mmr):
    return (entry_price * leverage) / (leverage - 1 + mmr * leverage)
```

---

## 5. Parte 3: Implementacion

### 5.1 Arquitectura de Streaming

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Output type

class Tick(ABC, Generic[T, R]):
    @abstractmethod
    def on_tick(self, val: T) -> R:
        """Procesa un nuevo tick."""
        pass
```

### 5.2 Sliding Window

```python
from collections import deque

class DequeWindow(Tick[T, Optional[T]]):
    def __init__(self, n: int):
        self._data = deque(maxlen=n)

    def on_tick(self, val: T) -> Optional[T]:
        dropped = self._data[0] if self.is_full() else None
        self._data.append(val)
        return dropped

    def is_full(self) -> bool:
        return len(self._data) == self._data.maxlen
```

### 5.3 Streaming Log Returns

```python
class LogReturn(Tick[float, Optional[float]]):
    def __init__(self):
        self._window = NumpyWindow(2)

    def on_tick(self, val: float) -> Optional[float]:
        self._window.on_tick(val)
        if self._window.is_full():
            return np.log(self._window[1] / self._window[0])
        return None
```

### 5.4 Streaming Lags

```python
class LogReturnLags(Tick[float, torch.Tensor]):
    def __init__(self, no_lags: int):
        self._lags = DequeWindow(no_lags)
        self._log_return = LogReturn()

    def on_tick(self, val: float) -> Optional[torch.Tensor]:
        log_ret = self._log_return.on_tick(val)
        if log_ret is not None:
            self._lags.append_left(log_ret)
            if self._lags.is_full():
                return torch.tensor(self._lags.to_numpy())
        return None
```

### 5.5 Sistema de Trading

```python
@dataclass(frozen=True)
class Order:
    sym: str
    signed_qty: Decimal

@dataclass(frozen=True)
class Trade:
    sym: str
    signed_qty: Decimal
    price: Decimal
    pnl: Decimal

@dataclass
class Position:
    sym: str
    signed_qty: Decimal
    price: Decimal

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        entry_val = self.price * self.signed_qty
        exit_val = current_price * -self.signed_qty
        return entry_val + exit_val
```

### 5.6 Estrategia Completa

```python
class BasicTakerStrat(Strategy):
    def __init__(self, sym, model, log_return_lags, scale_factor):
        self.sym = sym
        self.model = model
        self.log_return_lags = log_return_lags
        self.scale_factor = scale_factor

    def on_tick(self, price: float, account: Account) -> List[Order]:
        X = self.log_return_lags.on_tick(price)
        if X is not None:
            with torch.no_grad():
                y_hat = self.model(X)
                return self._create_orders(y_hat, account, price)
        return []
```

---

## 6. Metricas de Evaluacion

### 6.1 Metricas Principales

| Metrica | Formula | Interpretacion |
|---------|---------|----------------|
| **Sharpe Ratio** | `mean(returns) / std(returns) * sqrt(periods)` | > 1 es bueno, > 2 excelente |
| **Win Rate** | `winning_trades / total_trades` | > 50% con buen risk/reward |
| **Expected Value** | `win_rate * avg_win + (1-win_rate) * avg_loss` | Debe ser positivo |
| **Max Drawdown** | `min(equity - cummax(equity))` | Maximo declive |

### 6.2 Factor de Anualizacion

```python
def sharpe_annualization_factor(interval, trading_days=365, hours_per_day=24):
    # Para crypto que opera 24/7
    if interval == '12h':
        periods_per_year = trading_days * (hours_per_day / 12)
    return np.sqrt(periods_per_year)
```

### 6.3 Autocorrelacion

```python
# Matriz de correlacion entre target y sus lags
corr_matrix = research.auto_reg_corr_matrx(df, target, max_lags)
```

---

## 7. Estrategias Adicionales

### 7.1 Mean Reversion

El modelo AR(3) captura mean reversion:
- Peso negativo en lag_1 indica: "si sube, predice que baja"
- Explota la tendencia de precios a revertir a la media

### 7.2 Momentum

- Pesos positivos en lags: "si sube, sigue subiendo"
- Funciona mejor en tendencias fuertes

### 7.3 Indicadores Tecnicos

Ver archivos de estrategias adicionales:
- `strategy_rsi.py` - Relative Strength Index
- `strategy_macd.py` - Moving Average Convergence Divergence
- `strategy_bollinger.py` - Bollinger Bands

---

## 8. Mejores Practicas

### 8.1 Reproducibilidad

```python
research.set_seed(42)  # SIEMPRE al inicio
```

### 8.2 Split Temporal

**NUNCA** usar train_test_split aleatorio para series temporales:
```python
# CORRECTO
train, test = research.timeseries_split(df, test_size=0.25)

# INCORRECTO
from sklearn.model_selection import train_test_split
train, test = train_test_split(df)  # NO!
```

### 8.3 Evitar Overfitting

- Usar modelos simples (lineales)
- Validar en out-of-sample
- Monitorear alpha decay

### 8.4 Risk Management

1. **Position Sizing**: Nunca arriesgar mas del 2% por trade
2. **Leverage**: Escalar con Sharpe ratio
3. **Stop Loss**: Implementar limites de perdida
4. **Diversificacion**: Multiples activos/estrategias

### 8.5 Consideraciones de Produccion

- **Market Impact**: Tamanos pequenos no mueven el mercado
- **Slippage**: Ejecucion real difiere del backtest
- **Funding Fees**: Costos de mantener posiciones apalancadas
- **Alpha Decay**: El edge se degrada con el tiempo

---

## Apendice: Funciones Clave de research.py

### Carga de Datos
- `load_ohlc_timeseries_range()` - Carga OHLC en rango de fechas
- `load_timeseries_range()` - Carga con agregaciones custom

### Feature Engineering
- `add_log_return_features()` - Agrega log returns y lags
- `add_lags()` - Agrega columnas de lag
- `log_return()` - Expresion Polars para log return

### Entrenamiento
- `batch_train_reg()` - Entrenamiento batch
- `timeseries_train_test_split()` - Split temporal
- `benchmark_reg_model()` - Benchmark con metricas

### Evaluacion
- `eval_model_performance()` - Metricas completas
- `model_trade_results()` - Resultados por trade
- `sharpe_annualization_factor()` - Factor de anualizacion

### Visualizacion
- `plot_column()` - Grafico de linea
- `plot_distribution()` - Histograma
- `plot_dyn_timeseries()` - Grafico interactivo

### Trading
- `add_compounding_trades()` - PnL con compounding
- `add_tx_fees()` - Agregar fees
- `add_equity_curve()` - Curva de equity
