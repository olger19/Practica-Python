import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from core.algorithms import SimulatedAnnealing, HillClimbing
from core.problems import TSP, NQueens

st.set_page_config(page_title="Optimizador SA vs HC", layout="wide")

st.title("Comparativa de Algoritmos de Optimizacion")

# --- SIDEBAR: Configuración de Hiperparámetros ---
st.sidebar.header("Configuracion General")
problema_choice = st.sidebar.selectbox("Selecciona el Problema", ["TSP", "N-Queens"])
n = st.sidebar.slider("Tamaño del Problema (n)", 8, 100, 20)

st.sidebar.header("Hiperparámetros SA")
t0 = st.sidebar.number_input("Temperatura Inicial (T0)", value=100.0)
alpha = st.sidebar.slider("Factor Enfriamiento (Alpha)", 0.80, 0.99, 0.95)
t_min = st.sidebar.number_input("Temperatura Mínima", value=0.01)

# --- EJECUCIÓN ---
if st.button("Ejecutar Comparativa"):
    col1, col2 = st.columns(2)
    
    # Instanciar problema
    problem = TSP(n) if problema_choice == "TSP" else NQueens(n)
    
    with col1:
        st.subheader("Hill Climbing")
        start_hc = time.time()
        state_hc, res_hc, history_hc = HillClimbing.run(problem) # Debes modificar tu lógica para devolver historial
        st.write(f"**Mejor Fitness:** {res_hc}")
        st.write(f"**Tiempo:** {time.time() - start_hc:.4f}s")

    with col2:
        st.subheader("Simulated Annealing")
        start_sa = time.time()
        state_sa, res_sa, history_sa = SimulatedAnnealing.run(problem, t0, alpha, t_min)
        st.write(f"**Mejor Fitness:** {res_sa}")
        st.write(f"**Tiempo:** {time.time() - start_sa:.4f}s")

    # --- GRÁFICA DE CONVERGENCIA ---
    st.divider()
    st.subheader("Gráfica de Convergencia (Fitness vs Iteraciones)")
    fig, ax = plt.subplots()
    ax.plot(history_hc, label="Hill Climbing", color="red")
    ax.plot(history_sa, label="Simulated Annealing", color="blue")
    ax.set_xlabel("Iteraciones")
    ax.set_ylabel("Fitness (Menor es mejor)")
    ax.legend()
    st.pyplot(fig)