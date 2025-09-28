

# 📖 Documentación de referencia: SHAP en el MVP

## 1. ¿Qué es SHAP?

**SHAP (SHapley Additive exPlanations)** es una **librería de Python** para explicar modelos de Machine Learning.
No es un modelo predictivo, sino una herramienta de **explicabilidad**.

Se basa en la **teoría de valores de Shapley** (teoría de juegos cooperativos), que asigna a cada **feature/variable** una contribución al resultado final de una predicción.

👉 **Pregunta que responde SHAP:**
**“¿Por qué el modelo predijo este resultado?”**

---

## 2. ¿Cómo funciona?

* Cada predicción del modelo se descompone en las contribuciones de cada variable.
* La suma de todas esas contribuciones = predicción final.

Ejemplo simple:

* Predicción de riesgo: **0.82**
* Variables que influyen:

  * `asistencia_clases baja` (+0.25 riesgo)
  * `pocos_horas_sueño` (+0.12 riesgo)
  * `puntaje_test_memoria alto` (−0.05 riesgo)
  * `intervenciones_previas` (−0.02 riesgo)

---

## 3. Ejemplo en código (con XGBoost)

```python
import shap
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset proxy (ejemplo con MMASD adaptado)
df = pd.read_csv("data/processed/mmasd_features.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamos modelo baseline
model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)
model.fit(X_train, y_train)

# Creamos el objeto SHAP
explainer = shap.TreeExplainer(model)

# Calculamos valores SHAP
shap_values = explainer.shap_values(X_test)

# --- Visualización global ---
shap.summary_plot(shap_values, X_test)

# --- Visualización local (ejemplo de un niño específico) ---
i = 0  # índice del primer caso en test
shap.force_plot(explainer.expected_value, shap_values[i,:], X_test.iloc[i,:])
```

---

## 4. Tipos de visualización

* **Global (summary plot):**
  Ordena variables por importancia en todo el dataset.
  Ejemplo: `asistencia_clases` > `puntaje_memoria` > `horas_sueño`.

* **Local (force plot):**
  Explica una predicción individual.
  Ejemplo:

  * La baja asistencia **empuja hacia mayor riesgo**.
  * La intervención terapéutica **empuja hacia menor riesgo**.

---

## 5. Importancia para el proyecto

1. **Transparencia:** evita la “caja negra” del modelo.
2. **Ética:** explicaciones claras, no diagnósticos automáticos.
3. **Roles diferenciados:**

   * **Familia:** “El riesgo aumentó porque la asistencia bajó”.
   * **Docente:** “El faltante de tareas fue clave en la predicción”.
   * **Terapeuta:** “Los últimos puntajes clínicos redujeron el riesgo”.

---

## 6. Conexión con el MVP

* Se aplicará **SHAP a los modelos entrenados (XGBoost)** para generar explicaciones:

  * **Locales** (para cada niño/niña).
  * **Globales** (para todo el dataset).
* Los resultados se mostrarán en **dashboards por rol** con lenguaje adaptado:

  * **Familia:** mensajes simples y claros.
  * **Docentes/Terapeutas:** gráficos con top-variables SHAP.

---

✅ En resumen:
**SHAP = librería de explicabilidad para ML**.
Se integra a tu MVP para dar **explicaciones globales y locales** de las predicciones, reforzando la confianza, ética y utilidad del sistema.

---
