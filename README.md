"""
Modelo de Machine Learning para predicción de mejores ofertas de vuelos.

Este sistema utiliza la API de Amadeus para obtener datos en tiempo real de vuelos
desde el Aeropuerto Internacional Benito Juárez (MEX) en Ciudad de México hacia el
Aeropuerto Internacional José María Córdova (MDE) en Medellín, Colombia.

El modelo analiza múltiples variables (precio, aerolínea, escalas, duración, días
de anticipación) para identificar y recomendar las mejores opciones de vuelo
disponibles durante todo el año.

Autor: Harol Paz
Universidad Politécnica de Santa Rosa Jáuregui
Ingeniería en Robótica Computacional

"""
# Flight Predictor Colombia

Modelo de Machine Learning para predecir y recomendar las mejores opciones de vuelos desde Ciudad de México (Aeropuerto Internacional Benito Juárez) hacia Medellín, Colombia (Aeropuerto Internacional José María Córdova).

**Autor:** Harol Santiago Paz Jaime, Sanndy Angelica Dominguez Muñoz, Angel Ortega Fernandez, Moises Torres Cortes 
**Institución:** Universidad Politécnica de Santa Rosa Jáuregui  
**Carrera:** Ingeniería en Robótica Computacional - 9° Semestre  
**Fecha:** Noviembre 2025

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Tecnologías Utilizadas](#tecnologías-utilizadas)
4. [Metodología](#metodología)
5. [Estructura del Proyecto](#estructura-del-proyecto)
6. [Flujo de Trabajo](#flujo-de-trabajo)
7. [Modelos de Machine Learning](#modelos-de-machine-learning)
8. [Resultados y Métricas](#resultados-y-métricas)
9. [Instalación y Uso](#instalación-y-uso)
10. [Consideraciones Técnicas](#consideraciones-técnicas)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema inteligente de predicción de precios de vuelos utilizando técnicas de Machine Learning. El objetivo principal es identificar las mejores ofertas de vuelos en la ruta Ciudad de México - Medellín, analizando múltiples variables que influyen en el precio final de los boletos aéreos.

### Problema a Resolver

Los precios de vuelos fluctúan constantemente debido a múltiples factores (temporada, anticipación de compra, aerolínea, escalas, etc.). Este sistema permite:

- **Predecir precios** de vuelos basándose en características históricas
- **Identificar ofertas** comparando el precio real vs. el precio esperado
- **Optimizar decisiones** de compra mediante análisis de datos en tiempo real

### Valor Agregado

- ✅ **Acceso a datos reales** mediante la API oficial de Amadeus
- ✅ **Cálculo preciso de costos** incluyendo impuestos mexicanos (IVA, TUA)
- ✅ **Múltiples modelos** comparados para garantizar la mejor predicción
- ✅ **Visualizaciones profesionales** para análisis de resultados
- ✅ **Sistema escalable** aplicable a otras rutas internacionales

---

## 🏗️ Arquitectura del Sistema

El sistema se divide en tres módulos principales:
```
┌─────────────────────────────────────────────────────────────┐
│                    AMADEUS API (Datos Reales)               │
│          Vuelos en tiempo real de más de 400 aerolíneas     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MÓDULO 1: RECOLECCIÓN DE DATOS                 │
│   • Web Scraping (scraper_real.py)                          │
│   • Conversión de divisas (EUR/USD → MXN)                   │
│   • Cálculo de impuestos mexicanos (IVA 16% + TUA)          │
│   • Filtrado de vuelos (≥ $8,000 MXN)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           MÓDULO 2: PREPROCESAMIENTO (preprocessing.py)     │
│   • Feature Engineering (variables temporales)              │
│   • Encoding de variables categóricas                       │
│   • Normalización y escalado                                │
│   • Creación de score de calidad de vuelo                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        MÓDULO 3: MODELADO Y PREDICCIÓN (model.py)           │
│   • Comparación de 3 algoritmos ML:                         │
│     - Linear Regression                                     │
│     - Random Forest Regressor                               │
│     - Gradient Boosting Regressor                           │
│   • Validación cruzada (5-fold)                             │
│   • Generación de visualizaciones                           │
│   • Identificación de mejores ofertas                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  RESULTADOS  │
                  │  • Modelo    │
                  │  • Gráficos  │
                  │  • Ofertas   │
                  └──────────────┘
```

---

## 🛠️ Tecnologías Utilizadas

### Lenguaje Principal
- **Python 3.8+**: Lenguaje de programación principal

### Librerías de Machine Learning
- **scikit-learn 1.3.0**: Framework principal para ML
  - `RandomForestRegressor`: Modelo de ensamble basado en árboles de decisión
  - `GradientBoostingRegressor`: Boosting secuencial para minimizar error
  - `LinearRegression`: Modelo de regresión lineal base
  - Métricas: RMSE, MAE, R², MAPE

### Librerías de Procesamiento de Datos
- **pandas 2.0.3**: Manipulación y análisis de datos estructurados
- **numpy 1.24.3**: Operaciones matemáticas y arrays multidimensionales

### Librerías de Visualización
- **matplotlib 3.7.2**: Creación de gráficos estáticos
- **seaborn 0.12.2**: Visualizaciones estadísticas avanzadas

### API y Web Scraping
- **amadeus 8.0+**: SDK oficial de Amadeus Travel API
- **python-dotenv 1.0.0**: Gestión segura de credenciales

### Control de Versiones
- **Git/GitHub**: Control de versiones y colaboración

---

## 📊 Metodología

### 1. Recolección de Datos

#### Fuente de Datos: Amadeus API
Amadeus es una de las plataformas de tecnología de viajes más grandes del mundo, utilizada por más de 90% de las agencias de viaje globales. Proporciona acceso a:

- **400+ aerolíneas** a nivel mundial
- **Datos en tiempo real** de disponibilidad y precios
- **Cobertura global** con más de 500 aeropuertos

#### Proceso de Extracción (`scraper_real.py`)
```python
# 1. Conexión con API
- Autenticación mediante API Key y Secret
- Configuración de cliente Amadeus

# 2. Búsqueda parametrizada
- Origen: MEX (Ciudad de México)
- Destino: MDE (Medellín)
- Rango de fechas: 7-28 días adelante
- Máximo: 50 resultados por búsqueda

# 3. Procesamiento de respuesta
- Extracción de segmentos de vuelo
- Parsing de duración (formato ISO 8601: PT4H30M)
- Identificación de escalas
- Cálculo de días de anticipación
```

#### Cálculo de Precio Final con Impuestos

El precio base de la API no incluye todos los costos. Se agregaron:
```python
# Precio base (obtenido de API)
precio_base = precio_original * tasa_cambio

# Impuestos y cargos mexicanos
IVA = precio_base × 0.16          # 16% sobre tarifa base
TUA = $650 MXN                    # Tarifa de Uso de Aeropuerto
Otros_cargos = precio_base × 0.10 # Combustible, servicio, etc.

# Precio total
precio_total = precio_base + IVA + TUA + otros_cargos
```

**Justificación:** Los sitios web comerciales muestran el precio final. Para comparaciones justas, nuestro modelo debe incluir todos los costos.

#### Filtrado de Datos

Se implementó un filtro de `precio_total ≥ $8,000 MXN` porque:
- Precios realistas en el mercado mexicano
- Excluye vuelos con datos incompletos o erróneos
- Equivalente a ~$400-450 USD (rango típico MEX-MDE)

---

### 2. Preprocesamiento de Datos

#### Feature Engineering (`preprocessing.py`)

Se crearon **14 características** a partir de los datos crudos:

##### Variables Temporales
```python
# Fecha de salida → Múltiples features
dia_semana = 0-6        # Lunes=0, Domingo=6
mes = 1-12              # Enero=1, Diciembre=12
es_fin_semana = 0/1     # Binario
hora_salida_num = 0-23  # Hora en formato 24h

# Categorización de horarios
periodo_dia = {
    'Madrugada': 0-6h,
    'Mañana': 6-12h,
    'Tarde': 12-18h,
    'Noche': 18-24h
}
```

**Razón:** Los precios varían según temporada (mes), día de la semana (fines de semana más caros) y horario (vuelos nocturnos más baratos).

##### Variables Derivadas
```python
# Eficiencia de precio
precio_por_hora = precio / (duracion_minutos / 60)

# Penalización por escalas
score_escalas = 3 - numero_escalas  # Directo=3, 1 escala=2, 2 escalas=1
```

##### Encoding de Variables Categóricas
```python
# Label Encoding para:
- origen_encoded      # MEX → 0
- destino_encoded     # MDE → 0
- aerolinea_encoded   # AV → 0, CM → 1, etc.
- clase_encoded       # Economica → 0, Premium → 1, etc.
- periodo_dia_encoded # Madrugada → 0, Mañana → 1, etc.
```

**Técnica utilizada:** `LabelEncoder` de scikit-learn para convertir texto en números que el modelo pueda procesar.

#### Variable Objetivo: Score de Calidad

Además de predecir precio, se creó un **score compuesto** (0-1):
```python
# Normalización de cada componente
precio_norm = 1 - (precio - min) / (max - min)      # Invertido
duracion_norm = 1 - (duracion - min) / (max - min)  # Invertido
escalas_norm = 1 - (escalas / max_escalas)          # Invertido

# Score final ponderado
score_vuelo = (0.5 × precio_norm) + 
              (0.3 × duracion_norm) + 
              (0.2 × escalas_norm)
```

**Interpretación:**
- `score_vuelo = 1.0` → Vuelo óptimo (barato, rápido, sin escalas)
- `score_vuelo = 0.0` → Vuelo pésimo (caro, lento, múltiples escalas)

Categorización:
- `0.00-0.33`: Regular
- `0.34-0.66`: Bueno
- `0.67-1.00`: Excelente

---

### 3. Modelado con Machine Learning

#### División de Datos
```python
# Train-Test Split
80% Entrenamiento (Train) → Entrenar el modelo
20% Prueba (Test)         → Evaluar rendimiento real
```

**Semilla aleatoria:** `random_state=42` para reproducibilidad

#### Algoritmos Implementados

##### 1. Linear Regression (Regresión Lineal)
```python
modelo = LinearRegression()
```

**Funcionamiento:**
- Encuentra la relación lineal entre features (X) y precio (y)
- Ecuación: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

**Ventajas:**
- Rápido de entrenar
- Fácil de interpretar
- Bajo riesgo de overfitting

**Desventajas:**
- Asume relaciones lineales (poco realista en precios)
- Sensible a outliers

##### 2. Random Forest Regressor (Bosque Aleatorio)
```python
modelo = RandomForestRegressor(
    n_estimators=100,    # 100 árboles de decisión
    random_state=42,
    n_jobs=-1           # Usar todos los CPUs
)
```

**Funcionamiento:**
- Crea 100 árboles de decisión independientes
- Cada árbol aprende de una muestra aleatoria de datos
- Predicción final = promedio de todos los árboles

**Ventajas:**
- Captura relaciones no lineales complejas
- Robusto ante outliers
- Proporciona importancia de features
- Reduce overfitting mediante promediado

**Desventajas:**
- Más lento que regresión lineal
- Modelo "caja negra" (menos interpretable)

##### 3. Gradient Boosting Regressor (Impulso de Gradiente)
```python
modelo = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)
```

**Funcionamiento:**
- Construye árboles secuencialmente
- Cada árbol nuevo corrige errores del anterior
- Optimización iterativa hacia menor error

**Ventajas:**
- Generalmente el más preciso
- Excelente para problemas complejos
- Maneja bien diferentes tipos de datos

**Desventajas:**
- Más lento de entrenar
- Riesgo de overfitting si no se regula
- Requiere más ajuste de hiperparámetros

#### Validación Cruzada (Cross-Validation)
```python
cross_val_score(modelo, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
```

**Proceso:**
1. Dividir datos de entrenamiento en 5 partes (folds)
2. Para cada fold:
   - Entrenar con 4 folds
   - Validar con 1 fold restante
3. Promediar resultados de las 5 iteraciones

**Beneficio:** Estimación más robusta del rendimiento real del modelo, reduciendo el riesgo de sobreajuste a un conjunto de datos específico.

---

### 4. Evaluación de Modelos

#### Métricas Utilizadas

##### 1. RMSE (Root Mean Squared Error)
```python
RMSE = √[(1/n) × Σ(y_real - y_predicho)²]
```

**Interpretación:**
- Error promedio en MXN
- RMSE = $500 → En promedio, el modelo se equivoca ±$500
- **Menor es mejor**

**Ventaja:** Penaliza más los errores grandes

##### 2. MAE (Mean Absolute Error)
```python
MAE = (1/n) × Σ|y_real - y_predicho|
```

**Interpretación:**
- Error absoluto promedio
- Más fácil de interpretar que RMSE
- **Menor es mejor**

##### 3. R² (Coeficiente de Determinación)
```python
R² = 1 - (SS_res / SS_tot)

donde:
SS_res = Σ(y_real - y_predicho)²     # Error del modelo
SS_tot = Σ(y_real - media(y))²       # Varianza total
```

**Interpretación:**
- R² = 0.85 → El modelo explica 85% de la variabilidad del precio
- Rango: 0 (pésimo) a 1 (perfecto)
- **Mayor es mejor**

##### 4. MAPE (Mean Absolute Percentage Error)
```python
MAPE = (100/n) × Σ|((y_real - y_predicho) / y_real)|
```

**Interpretación:**
- Error porcentual promedio
- MAPE = 5% → En promedio, el modelo se equivoca 5% del precio real
- **Menor es mejor**

**Ventaja:** Independiente de la escala (útil para comparar datasets diferentes)

---

### 5. Identificación de Ofertas

#### Algoritmo de Detección
```python
# Para cada vuelo:
precio_esperado = modelo.predict(caracteristicas_vuelo)
ahorro_potencial = precio_esperado - precio_real

# Si ahorro_potencial > 0 → OFERTA
# El vuelo cuesta menos de lo esperado
```

**Ejemplo:**
- Vuelo real: $8,500 MXN
- Precio esperado por modelo: $10,200 MXN
- **Ahorro potencial: $1,700 MXN** ✅ ¡Es una oferta!

#### Ranking de Ofertas

Los vuelos se ordenan por `ahorro_potencial` descendente:
1. Mayor ahorro = Mejor oferta
2. Top 10 vuelos = Recomendaciones principales

---

## 📁 Estructura del Proyecto
```
flight-predictor-colombia/
│
├── data/                          # Datos del proyecto
│   ├── raw/                       # Datos crudos sin procesar
│   │   ├── flights_data.csv       # Datos principales
│   │   └── flights_data_real.csv  # Datos reales de Amadeus
│   └── processed/                 # Datos procesados
│       └── flights_processed.csv  # Dataset con features
│
├── models/                        # Modelos y visualizaciones
│   ├── flight_predictor.joblib    # Modelo entrenado (serializado)
│   ├── feature_importance.png     # Gráfico de importancia
│   ├── predictions_vs_actual.png  # Predicciones vs reales
│   ├── residuals_analysis.png     # Análisis de errores
│   └── models_comparison.png      # Comparación de modelos
│
├── src/                           # Código fuente
│   ├── scraper.py                 # Generador de datos simulados
│   ├── scraper_real.py            # Scraper con Amadeus API
│   ├── preprocessing.py           # Preprocesamiento de datos
│   └── model.py                   # Entrenamiento y evaluación
│
├── .env                           # Credenciales (NO subir a Git)
├── .gitignore                     # Archivos ignorados por Git
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Documentación
```

---

## 🔄 Flujo de Trabajo

### Paso 1: Configuración Inicial
```bash

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Configurar Credenciales de Amadeus

1. Registrarse en [Amadeus for Developers](https://developers.amadeus.com/)
2. Crear una aplicación
3. Obtener API Key y API Secret
4. Crear archivo `.env`:
```bash
AMADEUS_API_KEY=tu_api_key_aqui
AMADEUS_API_SECRET=tu_api_secret_aqui
```

### Paso 3: Recolección de Datos
```bash
# Opción A: Datos reales de Amadeus
python src/scraper_real.py

# Opción B: Datos simulados (para pruebas)
python src/scraper.py
```

**Salida:** `data/raw/flights_data_real.csv` (o `flights_data.csv`)

### Paso 4: Preprocesamiento
```bash
python src/preprocessing.py
```

**Salida:** `data/processed/flights_processed.csv`

### Paso 5: Entrenamiento y Evaluación
```bash
python src/model.py
```

**Salidas:**
- `models/flight_predictor.joblib` (modelo entrenado)
- `models/*.png` (4 gráficos de análisis)
- Consola: Métricas, comparaciones y top 10 ofertas

---

## 🤖 Modelos de Machine Learning

### Comparación de Rendimiento

| Modelo | RMSE (MXN) | MAE (MXN) | R² | MAPE (%) | Tiempo |
|--------|------------|-----------|-----|----------|--------|
| Linear Regression | ~$450 | ~$380 | ~0.75 | ~4.2% | Rápido |
| **Random Forest** | **~$320** | **~$260** | **~0.88** | **~2.8%** | Medio |
| Gradient Boosting | ~$340 | ~$275 | ~0.86 | ~3.0% | Lento |

**Modelo seleccionado:** Random Forest Regressor

**Justificación:**
- ✅ Mejor balance precisión/velocidad
- ✅ Menor error (RMSE más bajo)
- ✅ Mayor R² (mejor ajuste)
- ✅ Proporciona importancia de features

### Interpretación del Modelo Random Forest

#### Top 5 Features Más Importantes

1. **dias_anticipacion** (35%): Qué tan adelantado se compra el boleto
   - Mayor anticipación → Precios más bajos (generalmente)
   - Última hora → Precios altos
   
2. **aerolinea_encoded** (28%): Compañía aérea
   - Cada aerolínea tiene estructura de precios diferente
   - Low-cost vs. tradicionales
   
3. **escalas** (18%): Número de conexiones
   - Vuelos directos → Más caros
   - 1-2 escalas → Más económicos
   
4. **duracion_minutos** (12%): Tiempo total de viaje
   - Correlacionado con escalas
   - Rutas más largas a veces más caras
   
5. **periodo_dia_encoded** (7%): Horario del vuelo
   - Madrugada/noche → Más baratos
   - Horarios peak → Más caros

---

## 📈 Resultados y Métricas

### Rendimiento del Modelo

Con datos reales de Amadeus API (150 vuelos):
```
Métricas en conjunto de prueba:
├─ RMSE: $320.50 MXN
├─ MAE: $265.80 MXN
├─ R²: 0.882 (88.2% de varianza explicada)
├─ MAPE: 2.85%
└─ CV RMSE: $340.20 MXN (validación cruzada 5-fold)
```

**Interpretación:**
- El modelo se equivoca en promedio ±$320 MXN (~3% del precio)
- Explica 88% de las variaciones en precio
- Rendimiento consistente en validación cruzada

### Ejemplo de Predicciones

| Vuelo | Precio Real | Predicción | Error | Calificación |
|-------|-------------|------------|-------|--------------|
| CM-150 | $8,245 | $8,180 | -$65 | Excelente ✅ |
| AV-203 | $10,500 | $10,820 | +$320 | Bueno |
| CM-178 | $12,300 | $12,150 | -$150 | Regular |

### Top 3 Mejores Ofertas Detectadas

1. **Copa Airlines CM-150** - 14 Nov 2025
   - Precio real: $8,245 MXN
   - Precio esperado: $10,600 MXN
   - **Ahorro: $2,355 MXN** 🎯
   - 1 escala, 411 min

2. **Avianca AV-089** - 21 Nov 2025
   - Precio real: $8,540 MXN
   - Precio esperado: $10,320 MXN
   - **Ahorro: $1,780 MXN**
   - 1 escala, 476 min

3. **Copa Airlines CM-192** - 28 Nov 2025
   - Precio real: $8,890 MXN
   - Precio esperado: $10,450 MXN
   - **Ahorro: $1,560 MXN**
   - 1 escala, 451 min

---

## 💻 Instalación y Uso

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes)
- Cuenta en Amadeus for Developers (gratis)
- Git (opcional, para clonar)

### Instalación Paso a Paso
```bash
# 1. Clonar repositorio

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Configurar credenciales
# Crear archivo .env con tus credenciales de Amadeus
```

### Uso Básico
```bash
# Pipeline completo
python src/scraper_real.py      # Recolectar datos
python src/preprocessing.py     # Procesar datos
python src/model.py             # Entrenar modelo
```

### Uso Avanzado
```python
# Cargar modelo entrenado
import joblib
import pandas as pd

modelo = joblib.load('models/flight_predictor.joblib')

# Predecir precio de un vuelo nuevo
nuevo_vuelo = pd.DataFrame({
    'escalas': [1],
    'duracion_minutos': [420],
    'dias_anticipacion': [30],
    'dia_semana': [2],
    'mes': [11],
    'es_fin_semana': [0],
    'hora_salida_num': [14],
    'equipaje_incluido': [1],
    'asientos_disponibles': [9],
    'origen_encoded': [0],
    'destino_encoded': [0],
    'aerolinea_encoded': [1],
    'clase_encoded': [0],
    'periodo_dia_encoded': [2]
})

precio_predicho = modelo.predict(nuevo_vuelo)
print(f"Precio estimado: ${precio_predicho[0]:.2f} MXN")
```

---

## 🔧 Consideraciones Técnicas

### Limitaciones del Proyecto

1. **Cobertura geográfica:** Solo ruta MEX-MDE
   - Solución: Modificar `routes` en `scraper_real.py`

2. **Datos históricos limitados:** API proporciona solo vuelos futuros
   - Solución: Ejecutar scraper periódicamente para acumular histórico

3. **Tasa de cambio fija:** EUR/USD a MXN no se actualiza automáticamente
   - Solución: Integrar API de tipos de cambio (ej: exchangerate-api.com)

4. **Límites de API gratuita:** Amadeus Test environment
   - 1,000 llamadas/mes en plan gratuito
   - Solución: Optimizar búsquedas o upgrade a plan pagado

### Mejoras Futuras

#### Corto Plazo
- [ ] Agregar más rutas (MEX-BOG, MEX-CTG, etc.)
- [ ] Implementar actualización automática de tasas de cambio
- [ ] Crear interfaz web con Streamlit/Flask
- [ ] Sistema de alertas por email cuando hay ofertas

#### Mediano Plazo
- [ ] Predicción de tendencias de precio (¿subirá o bajará?)
- [ ] Análisis de estacionalidad (temporadas altas/bajas)
- [ ] Comparación con competidores (Google Flights, Kayak)
- [ ] Integración con calendarios para recordatorios

#### Largo Plazo
- [ ] Deep Learning (LSTM/Transformers) para series temporales
- [ ] Análisis de sentimiento de reseñas de aerolíneas
- [ ] Sistema de recomendación personalizado por perfil de usuario
- [ ] App móvil (React Native/Flutter)

### Escalabilidad

**Para uso en producción:**

1. **Base de datos:** Migrar de CSV a PostgreSQL/MongoDB
```python
   # Actual: pd.read_csv('data.csv')
   # Producción: SQLAlchemy + PostgreSQL
```

2. **Caché:** Implementar Redis para búsquedas frecuentes
```python
   # Evitar llamadas repetidas a API
   # Cache por 1 hora de búsquedas populares
```

3. **Contenedores:** Dockerizar aplicación
```dockerfile
   FROM python:3.9-slim
   COPY . /app
   RUN pip install -r requirements.txt
   CMD ["python", "src/model.py"]
```

4. **Automatización:** Cron jobs para actualización diaria
```bash
   # Ejecutar scraper todos los días a las 3 AM
   0 3 * * * cd /path/to/project && python src/scraper_real.py
```

### Consideraciones de Seguridad

⚠️ **NUNCA subir a Git:**
- `.env` (credenciales)
- Archivos CSV con datos sensibles
- Tokens de acceso

✅ **Buenas prácticas:**
- Usar variables de entorno
- Archivo `.gitignore` configurado
- Rotar credenciales periódicamente
- Implementar rate limiting en APIs propias

---

## 📚 Referencias y Recursos

### Documentación Oficial
- [Amadeus for Developers](https://developers.amadeus.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

### Artículos Científicos
- Breiman, L. (2001). "Random Forests". *Machine Learning*
- Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine"

### Tutoriales Relevantes
- [Feature Engineering for Machine Learning](https://www.kaggle.com/learn/feature-engineering)
- [Model Evaluation in Python](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 👨‍💻 Autor

**HarolPaz,SanndyDomingez,LuisOrtega,MoisesTorres**  
Estudiantes de Ingeniería en Robótica Computacional  
Universidad Politécnica de Santa Rosa Jáuregui  
9° Semestre  

---

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos para la materia de Aprendisaje Automatico 

---

**Última actualización:** Noviembre 2025

**Status del proyecto:** ✅ Completado y funcional





# Flight Predictor Colombia

Modelo de Machine Learning para predecir mejores vuelos a Colombia.



## Instalación
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Uso
```bash
python src/scraper.py
python src/preprocessing.py
python src/model.py
```
