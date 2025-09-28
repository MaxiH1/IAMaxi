import os
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="MVP Inteligente", layout="wide")

@st.cache_resource
def load_model():
    model_path = Path("artifacts/model.pkl")
    if not model_path.exists():
        st.warning("No se encontró artifacts/model.pkl. Entrenar en Paso 5.")
        return None
    return joblib.load(model_path)

def predict_case(model, X_df):
    # X_df: DataFrame con columnas en el mismo orden de entrenamiento
    y_prob = model.predict_proba(X_df)[:, 1]
    return float(y_prob[0])

def main():
    st.sidebar.title("MVP Cloud")
    rol = st.sidebar.selectbox("Rol", ["Familia", "Docente", "Terapeuta"])
    st.title("Plataforma clínica-educativa · MVP Cloud")

    model = load_model()

    st.subheader("Demo — Cargar caso (input mínimo)")
    # Placeholder: en el Paso 7 conectamos a un loader real
    feat_cols = ["success_rate_3", "time_median_5", "delta_score"]  # ejemplo
    vals = []
    for c in feat_cols:
        vals.append(st.number_input(c, value=0.0))
    X_case = pd.DataFrame([vals], columns=feat_cols)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Predecir"):
            if model is None:
                st.error("Falta modelo. Completá Paso 5 (entrenamiento).")
            else:
                score = predict_case(model, X_case)
                st.metric("Probabilidad de riesgo/mejora", f"{score:.2%}")

    st.divider()
    st.subheader("Explicación (SHAP)")
    st.caption("Se implementa en Paso 6: top-5 features por caso y summary global.")

    st.divider()
    st.subheader("Recomendaciones automáticas (RAG)")
    st.caption("Se implementa en Paso 9–11 (Qdrant + HF) sin preguntas del usuario, adaptadas por rol.")

if __name__ == "__main__":
    main()
