# Metamodelo de Predicción de Dirección de Precios de Commodities

**Prueba Técnica - Modelador Junior**

## 📑 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Datos](#datos)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Conclusiones](#conclusiones)
- [Instalación y Uso](#instalación-y-uso)
- [Dependencias](#dependencias)

---

## Descripción del Proyecto

Desarrollo de un metamodelo de clasificación que predice la dirección (subida/bajada) de precios de commodities utilizando las predicciones de 3 modelos base optimizados (AA6KBD, IPBG4J, OBONV1), específicamente para horizontes de 4 semanas.

### 🎯 Resultados Clave

- **Modelo**: Random Forest Classifier
- **Accuracy**: 82.35% (14/17 predicciones correctas)
- **F1-Score**: 76.92%
- **Recall**: 100% (captura todas las subidas)
- **Dataset**: 56 registros, 28 features, 0 valores nulos
- **Modelos base**: 3 de 5 (selección por cobertura >50%)

### Problema de Negocio

Dado un conjunto de predicciones de múltiples modelos sobre el precio futuro de un commodity, determinar si el precio real será mayor o menor que el precio actual, permitiendo tomar decisiones de trading informadas basadas en el consenso y dispersión de las predicciones.

### Variable Objetivo

- **`direction_real = 1`**: Si el precio SUBE (precio_futuro > precio_actual)
- **`direction_real = 0`**: Si el precio BAJA o se mantiene (precio_futuro ≤ precio_actual)

---

## Estructura del Proyecto

```
/Users/christian/Prueba/
├── notebooks/                          # Jupyter notebooks organizados
│   ├── 01_analisis_exploratorio.ipynb # EDA completo
│   ├── 02_preparacion_datos.ipynb     # Limpieza y preparación
│   ├── 03_feature_engineering.ipynb   # Creación de features
│   ├── 04_modelado_clasificacion.ipynb # Entrenamiento y evaluación
│   └── 05_evaluacion_resultados.ipynb # Análisis final y conclusiones
│
├── data/                               # Datos del proyecto
│   ├── raw/                            # Datos originales
│   │   └── ReportDataALL_20250404 (1).xlsx
│   ├── processed/                      # Datos procesados
│   │   ├── dataset_completo.csv        # 56 registros, 8 columnas (3 modelos)
│   │   └── dataset_final.csv           # 56 registros, 33 columnas (28 features)
│   └── results/                        # Modelos entrenados
│       ├── modelo_final.pkl            # Random Forest entrenado (134KB)
│       ├── scaler.pkl                  # StandardScaler (2.1KB)
│       └── imputer.pkl                 # SimpleImputer (1.5KB) [legacy]
│
├── outputs/                            # Resultados y visualizaciones
│   └── figures/                        # Gráficos generados (6 figuras)
│       ├── 01_serie_temporal_precios.png
│       ├── 02_distribucion_horizontes.png
│       ├── 03_analisis_errores_modelos.png
│       ├── 04_cambios_precio.png
│       ├── 05_correlacion_features.png
│       └── 06_feature_importance.png
│
├── Prueba Técnica - Modelador Junior.pdf  # Documento de requisitos
├── README.md                           # Este archivo (documentación completa)
├── requirements.txt                    # Dependencias Python
├── .gitignore                          # Archivos ignorados por Git
└── .dockerignore                       # Archivos ignorados por Docker
```

---

## Datos

### Fuente de Datos

**Archivo**: `ReportDataALL_20250404.xlsx`

#### Hoja "Real"

- **66 registros** de precios históricos
- **Período**: 2023-12-21 a 2025-04-03
- **Commodity**: Granular (fertilizante)
- **Rango de precios**: $201.69 - $392.39

#### Hoja "Predicted"

- **1,704 registros** de predicciones
- **5 modelos originales**: AA6KBD, HFWV8N, IPBG4J, LFHXNV, OBONV1
- **3 modelos seleccionados**: AA6KBD, IPBG4J, OBONV1 (cobertura >50%)
- **Múltiples horizontes temporales**: 7-56 días

### Dataset Final

Después del procesamiento y optimización:

- **56 registros** con predicciones a 4 semanas (27-29 días)
- **3 modelos base** seleccionados por cobertura de datos
- **28 features** creadas mediante feature engineering
- **0 valores nulos** gracias a imputación inteligente
- **Dataset limpio**: Split 70% train (39) / 30% test (17)

---

## Metodología

### 1. Análisis Exploratorio (EDA)

**Notebook 01**: `01_analisis_exploratorio.ipynb`

- Análisis de distribución temporal de precios
- Comportamiento de cada modelo base
- Identificación de patrones y correlaciones
- Análisis de errores por modelo
- **Visualizaciones**: 3 GraficoS

**Hallazgos clave**:

- Precios con volatilidad moderada (CV = 20%)
- Variabilidad significativa entre modelos
- Oportunidad clara para metamodelo

### 2. Preparación de Datos

**Notebook 02**: `02_preparacion_datos.ipynb`

#### Proceso de Optimización

**Análisis de Cobertura por Modelo:**

- AA6KBD: 77% cobertura ✓
- IPBG4J: 77% cobertura ✓
- OBONV1: 100% cobertura ✓
- HFWV8N: 46% cobertura ✗ (eliminado)
- LFHXNV: 30% cobertura ✗ (eliminado)

**Decisión Técnica:**
Se eliminaron HFWV8N y LFHXNV por baja cobertura (<50%), lo que habría generado imputación masiva y sesgo en el metamodelo.

#### Pasos Realizados

- **Filtrado**: Horizonte temporal de 4 semanas (27-29 días)
- **Selección**: 3 modelos con cobertura >50%
- **Pivoteo**: Convertir formato largo a ancho (1 columna por modelo)
- **Imputación inteligente**: Forward fill + mediana por modelo
- **Unión**: Merge con precios reales (actual y futuro)
- **Variable objetivo**: Creación de `direction_real`
- **Validaciones**: 0 nulos, sin duplicados

**Resultado**: Dataset limpio de 56 registros con 3 predicciones por fila (0 valores nulos)

### 3. Feature Engineering

**Notebook 03**: `03_feature_engineering.ipynb`

#### Features creadas (28 totales):

**A. Features por Modelo** (9 features = 3 × 3 modelos)

- Cambio absoluto: `predicción - precio_actual`
- Cambio porcentual: `(cambio / precio_actual) × 100`
- Dirección predicha: `1` si predice subida, `0` si no

**B. Features Agregadas** (6 features)

- `avg_prediction`: Media de predicciones
- `median_prediction`: Mediana
- `std_prediction`: Desviación estándar
- `min_prediction`, `max_prediction`: Valores extremos
- `range_prediction`: Rango de predicciones

**C. Features de Consenso** (5 features)

- `consensus_direction`: Dirección mayoritaria
- `consensus_bullish`: Cantidad de modelos que predicen subida
- `consensus_bearish`: Cantidad que predicen bajada
- `consensus_mixed`: Indicador de desacuerdo
- `direction_agreement`: Grado de acuerdo (0-1)

**D. Features de Dispersión** (5 features)

- `avg_change`: Cambio promedio predicho
- `avg_pct_change`: Cambio porcentual promedio
- `std_change`, `std_pct_change`: Dispersión de cambios
- `prediction_confidence`: Confianza (1 / (1 + std))

**E. Features de Diferencias entre Modelos** (3 features)

- Diferencias entre pares de modelos para capturar divergencias

**Análisis de correlación**: Identificación de features más prometedoras

#### Visualizaciones Generadas

1. **01_serie_temporal_precios.png**: Evolución temporal de precios reales
2. **02_distribucion_horizontes.png**: Distribución de horizontes de predicción
3. **03_analisis_errores_modelos.png**: Análisis de errores por modelo base
4. **04_cambios_precio.png**: Distribución de cambios de precio
5. **05_correlacion_features.png**: Matriz de correlación entre features
6. **06_feature_importance.png**: Top 5 features más importantes (generado en notebook 04)

### 4. Modelado y Clasificación

**Notebook 04**: `04_modelado_clasificacion.ipynb`

#### Configuración

- **Split**: 70% entrenamiento, 30% test (sin validación cruzada, según requisitos)
- **Orden temporal**: Sin shuffle para mantener consistencia temporal
- **Preprocesamiento**: StandardScaler para normalización

#### Modelos Evaluados

**Modelo 1: Logistic Regression**

- Regularización L2
- `max_iter=1000`
- Coeficientes como feature importance

**Modelo 2: Random Forest**

- `n_estimators=100`
- `max_depth=5`
- Feature importances nativas

#### Selección del Modelo

- **Criterio**: F1-Score (balance entre precisión y recall)
- **Modelo seleccionado**: [Se determina en ejecución]

#### Feature Importance - Top 5

Identificación y visualización de las 5 variables más importantes para el modelo final.

### 5. Evaluación y Resultados

**Notebook 05**: `05_evaluacion_resultados.ipynb`

#### Métricas de Evaluación (Requisitos)

- **Accuracy**: Proporción de aciertos generales
- **Precision**: Precisión en predicciones positivas
- **Recall**: Capacidad de detectar subidas
- **F1-Score**: Media armónica (balance)
- **R2 Score**: Capacidad explicativa

#### Análisis de Performance

- Matriz de confusión detallada
- Análisis de errores (FP, FN)
- Curvas ROC y Precision-Recall
- Interpretabilidad del modelo

---

## Resultados

### Modelo Final

**Random Forest Classifier** (Seleccionado por mejor F1-Score)

#### Métricas de Performance (Test Set: 17 registros)

- **Accuracy**: 82.35%
- **Precision**: 62.50%
- **Recall**: 100.00%
- **F1-Score**: 76.92%
- **R2 Score**: 0.1649

#### Matriz de Confusión

```
              Predicho
              Baja  Sube
Real  Baja     9     3
      Sube     0     5
```

**Interpretación:**

- El modelo captura TODAS las subidas reales (Recall 100%)
- Clasifica correctamente 14 de 17 casos (82.35% accuracy)
- Genera 3 falsos positivos (predice subida cuando baja)
- 0 falsos negativos (no pierde ninguna oportunidad de subida)
- Mejor desempeño que Logistic Regression

### Top 5 Features Más Importantes

1. **avg_pct_change** (0.1251): Cambio porcentual promedio predicho por los 3 modelos
2. **AA6KBD_pct_change** (0.1164): Cambio porcentual predicho por el modelo AA6KBD
3. **avg_change** (0.1051): Cambio absoluto promedio predicho
4. **IPBG4J_change** (0.1044): Cambio absoluto predicho por el modelo IPBG4J
5. **IPBG4J_pct_change** (0.0839): Cambio porcentual predicho por el modelo IPBG4J

**Insight clave:** Las features de cambio porcentual y las predicciones de modelos individuales (especialmente AA6KBD e IPBG4J) son las más determinantes para predecir la dirección del precio.

---

## Conclusiones

### Resumen Ejecutivo

1. **Modelo exitoso con 82.35% de accuracy**: El Random Forest logró predecir correctamente 14 de 17 movimientos de precio en el test set, con un Recall perfecto (100%) para detectar subidas y F1-Score de 76.92%.
2. **Optimización de modelos base crucial**: La eliminación de modelos con baja cobertura (HFWV8N, LFHXNV) y la imputación inteligente permitieron un dataset limpio de 56 registros con 0 valores nulos, sin comprometer la calidad predictiva.
3. **Features de cambio porcentual dominan**: Las 5 características más importantes están relacionadas con cambios porcentuales y predicciones de modelos individuales (AA6KBD, IPBG4J), validando la importancia de capturar tanto el consenso como las señales individuales.
4. **Trade-off Precision-Recall favorable**: Aunque la precisión es moderada (62.5%), el Recall del 100% significa que el modelo NO PIERDE ninguna oportunidad de subida, ideal para estrategias conservadoras de trading donde es crítico no perder oportunidades alcistas.

### Limitaciones Identificadas

- **Dataset moderado**: 56 registros (39 train, 17 test) limitan la capacidad de generalización
- **Un solo commodity**: Resultados específicos para fertilizante Granular
- **Horizonte fijo**: Solo predicciones a 4 semanas
- **3 modelos base**: Reducción de 5 a 3 modelos por cobertura de datos

### Próximos Pasos Recomendados

1. **Recolección de datos**: Ampliar el histórico para mejorar robustez estadística
2. **Multi-commodity**: Expandir a otros commodities para validar generalización
3. **Ensemble avanzado**: Probar XGBoost, LightGBM o stacking con validación cruzada
4. **Múltiples horizontes**: Adaptar a predicciones de 1, 2, 3 y 4 semanas simultáneamente
5. **Monitoreo continuo**: Implementar pipeline de reentrenamiento automático con nuevos datos

---

## Instalación y Uso

### Instalación Local

#### Requisitos Previos

- Python 3.9+
- pip o conda

#### Instalación

```bash
# Clonar repositorio
git clone git@github.com:Christ02/precios_de_commodities.git
cd precios_de_commodities

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

#### Ejecución

```bash
# Iniciar Jupyter Notebook
jupyter notebook notebooks/

# O ejecutar notebooks en orden:
# 1. 01_analisis_exploratorio.ipynb
# 2. 02_preparacion_datos.ipynb
# 3. 03_feature_engineering.ipynb
# 4. 04_modelado_clasificacion.ipynb
# 5. 05_evaluacion_resultados.ipynb
```

### Uso del Modelo Entrenado

```python
# Cargar modelo y scaler guardados
import joblib
import pandas as pd

# Cargar artefactos (134KB modelo + 2.1KB scaler)
modelo = joblib.load('data/results/modelo_final.pkl')
scaler = joblib.load('data/results/scaler.pkl')

# Cargar datos (debe tener las 28 features)
df = pd.read_csv('data/processed/dataset_final.csv')

# Separar features (28 columnas) de metadata (5 columnas)
exclude = ['date_requested', 'date_prediction', 'direction_real', 
           'value_at_request', 'value_at_prediction']
X = df[[c for c in df.columns if c not in exclude]]

# Escalar y predecir
X_scaled = scaler.transform(X)
predicciones = modelo.predict(X_scaled)
probabilidades = modelo.predict_proba(X_scaled)

# Ejemplo de predicción individual
idx = 0
print(f"Fecha: {df.iloc[idx]['date_requested']} -> {df.iloc[idx]['date_prediction']}")
print(f"Precio actual: ${df.iloc[idx]['value_at_request']:.2f}")
print(f"Dirección predicha: {'SUBE ↑' if predicciones[idx] == 1 else 'BAJA ↓'}")
print(f"Confianza: {max(probabilidades[idx]):.2%}")
print(f"Dirección real: {'SUBE ↑' if df.iloc[idx]['direction_real'] == 1 else 'BAJA ↓'}")
```

---

## Dependencias

Ver `requirements.txt` para lista completa. Dependencias del proyecto:

```
pandas==2.1.4          # Manipulación de datos
numpy==1.26.2          # Operaciones numéricas
matplotlib==3.8.2      # Visualizaciones
seaborn==0.13.0        # Gráficos estadísticos
scikit-learn==1.3.2    # Machine Learning
openpyxl==3.1.2        # Lectura de Excel
jupyter==1.0.0         # Notebooks interactivos
joblib==1.3.2          # Serialización de modelos
```

---

## Estructura de Commits

El proyecto sigue una estructura de commits organizada y profesional:

1. `Initial commit: project structure and configuration`
2. `Agregar análisis exploratorio de datos completo`
3. `Agregar preparación y limpieza de datos`
4. `Actualizar notebooks 01 y 02: Limpieza y optimización`
5. `Agregar ingeniería de características (notebook 03)`
6. `Agregar modelado y clasificación (notebook 04)` [Pendiente]
7. `Agregar evaluación final y conclusiones (notebook 05)` [Pendiente]
8. `Actualizar README con resultados completos` [Pendiente]

## Autor

**Christian Barrios**
Modelador Junior Candidate

- Email: barriosc31@gmail.com | christianbarrios@ufm.edu
- LinkedIn: Christian Barrios
