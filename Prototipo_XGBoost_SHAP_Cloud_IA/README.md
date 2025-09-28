# MVP Cloud — Plataforma clínica-educativa

## Estructura
- `src/data/mmasd_adapter.py`: adapta MMASD → `data/processed/mmasd_features.csv`
- `train.py` / `predict.py`: entrenamiento e inferencia XGBoost
- `src/explainer.py`: SHAP (explicabilidad)
- `src/rag/*`: RAG automático (recomendaciones con citas)
- `streamlit_app.py`: UI por rol (Familia/Docente/Terapeuta)

## Pasos
1) Cargar MMASD en `data/raw/MMASD/`
2) Ejecutar adapter → `data/processed/mmasd_features.csv`
3) Entrenar → `artifacts/model.pkl`
4) Ejecutar app Streamlit
