import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import os
from pathlib import Path

# =====================
# CONFIGURACIÓN STREAMLIT
# =====================
st.set_page_config(
    page_title="Predicción de Saturación Hospitalaria",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Predicción de Saturación Hospitalaria en España")
st.markdown("""
Este dashboard muestra la **predicción de pacientes hospitalizados** en España 
basada en datos abiertos de [Our World in Data](https://ourworldindata.org/covid-hospitalizations).
""")

# =====================
# CARGAR DATOS
# =====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hospital_spain_clean.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# =====================
# CARGAR MODELO
# =====================
@st.cache_resource
def load_model():
    return joblib.load("../src/hospital_model.pkl")

model = load_model()

# =====================
# SELECCIÓN DE PARÁMETROS
# =====================
st.sidebar.header("⚙️ Configuración")
future_days = st.sidebar.slider("Días a predecir", min_value=7, max_value=60, value=14, step=7)

# =====================
# ENTRENAR / PREDICCION
# =====================
df_prophet = df[['date', 'hosp_patients']].rename(columns={'date': 'ds', 'hosp_patients': 'y'})
future = model.make_future_dataframe(periods=future_days)
forecast = model.predict(future)

# =====================
# GRÁFICOS
# =====================
st.subheader("📈 Predicción de hospitalizados")
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("📊 Evolución histórica")
fig_hist = px.line(df, x="date", y=["hosp_patients", "icu_patients"],
                   labels={"value": "Pacientes", "date": "Fecha", "variable": "Tipo"},
                   title="Evolución histórica de hospitalizados y UCI")
st.plotly_chart(fig_hist, use_container_width=True)

# =====================
# TABLA DE PREDICCIONES
# =====================
st.subheader("📋 Predicciones detalladas")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_days))

# =====================
# PIE DE PÁGINA
# =====================
st.markdown("---")
st.markdown("📌 Proyecto desarrollado por **Francisco Javier Gómez Pulido**")
st.markdown("📊 Datos: Our World in Data")
st.markdown("📜 Perfil de Linkedin: www.linkedin.com/in/frangomezpulido")
st.markdown("🔗 Repositorio de GitHub: https://github.com/fragompul/hospital_capacity_prediction.git")
