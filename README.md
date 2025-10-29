# Metamodelo de Predicción de Precios de Commodities

**Prueba Técnica - Modelador Junior**

---

## Descripción

Metamodelo de clasificación que predice la dirección (subida/bajada) de precios de commodities usando predicciones de 3 modelos base (AA6KBD, IPBG4J, OBONV1) para un horizonte de 4 semanas.

### Resultados Principales

- **Modelo**: Random Forest Classifier
- **Accuracy**: 76.47% (13/17 correctas)
- **F1-Score**: 60%
- **Dataset**: 56 registros, 28 features

---

## Estructura del Proyecto

```
notebooks/
├── 01_analisis_exploratorio.ipynb     # EDA
├── 02_preparacion_datos.ipynb         # Limpieza y filtrado
├── 03_feature_engineering.ipynb       # Creación de features
├── 04_modelado_clasificacion.ipynb    # Entrenamiento
└── 05_evaluacion_resultados.ipynb     # Evaluación final

data/
├── raw/ReportDataALL_20250404.xlsx    # Datos originales
├── processed/
│   ├── dataset_completo.csv           # 56 registros, 3 modelos
│   └── dataset_final.csv              # 56 registros, 28 features
└── results/
    ├── modelo_final.pkl               # Random Forest
    └── scaler.pkl                     # StandardScaler

outputs/figures/                        # 6 visualizaciones
```

---

## Metodología

### 1. Datos

- **Origen**: Excel con precios reales y predicciones de 5 modelos
- **Filtrado**: Horizonte de 4 semanas (27-29 días)
- **Selección**: 3 modelos con cobertura >50% (eliminados HFWV8N y LFHXNV)
- **Resultado**: 56 registros sin valores nulos

### 2. Features (28 totales)

- Por modelo: cambio absoluto, cambio %, dirección (9 features)
- Agregadas: mean, median, std, min, max, range (6 features)
- Consenso: dirección mayoritaria, acuerdo, bullish/bearish (5 features)
- Dispersión: avg_change, avg_pct_change, std, confidence (5 features)
- Diferencias entre modelos (3 features)

### 3. Modelado

- **Split**: 70% train (39) / 30% test (17), sin shuffle
- **Modelos**: Logistic Regression vs Random Forest
- **Preprocesamiento**: StandardScaler
- **Selección**: Random Forest (mejor F1-Score)

### 4. Evaluación

**Métricas**:

- Accuracy: 76.47%
- Precision: 60%
- Recall: 60%
- F1-Score: 60%

**Matriz de Confusión**:

```
              Predicho
              Baja  Sube
Real  Baja     10     2
      Sube     2     3
```

**Top 5 Features**:

1. avg_pct_change (12.51%)
2. AA6KBD_pct_change (11.64%)
3. avg_change (10.51%)
4. IPBG4J_change (10.44%)
5. IPBG4J_pct_change (8.39%)

---

## Conclusiones

### Resumen

- Modelo balanceado con 76.47% de accuracy
- Features de cambio porcentual y consenso son las más importantes
- Eliminación de modelos con baja cobertura fue crucial para dataset limpio

### Limitaciones

- Dataset pequeño (56 registros) limita generalización
- Un solo commodity (fertilizante Granular)
- Horizonte temporal fijo (4 semanas)

### Próximos Pasos

1. Ampliar dataset (objetivo: >200 registros)
2. Expandir a múltiples commodities
3. Probar XGBoost/LightGBM
4. Optimizar hiperparámetros
5. Implementar múltiples horizontes temporales

---

## Instalación

```bash
# Clonar repositorio
git clone git@github.com:Christ02/precios_de_commodities.git
cd precios_de_commodities

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebooks
jupyter notebook notebooks/
```

---

## Uso del Modelo

```python
import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load('data/results/modelo_final.pkl')
scaler = joblib.load('data/results/scaler.pkl')

# Cargar datos
df = pd.read_csv('data/processed/dataset_final.csv')
exclude = ['date_requested', 'date_prediction', 'direction_real', 
           'value_at_request', 'value_at_prediction']
X = df[[c for c in df.columns if c not in exclude]]

# Predecir
X_scaled = scaler.transform(X)
predicciones = modelo.predict(X_scaled)
probabilidades = modelo.predict_proba(X_scaled)
```

---

## Dependencias

```
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn>=1.3.2
openpyxl==3.1.2
jupyter==1.0.0
joblib==1.3.2
```

---

## Autor

**Christian Barrios**
Email: barriosc31@gmail.com | christianbarrios@ufm.edu
GitHub: [@Christ02](https://github.com/Christ02)

---

**Última actualización**: Octubre 2025
