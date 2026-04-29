import itertools
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.algorithms import HillClimbing, SimulatedAnnealing
from core.problems import NQueens, TSP


st.set_page_config(page_title="IA: SA vs Hill Climbing", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def create_problem(problem_name, n_size, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    return TSP(n_size) if problem_name == "TSP (Viajero)" else NQueens(n_size)


def run_hill_climbing(problem, seed, callback=None):
    random.seed(seed)
    return HillClimbing.run(problem, callback=callback)


def run_simulated_annealing(problem, t0, alpha, t_min, seed, callback=None):
    random.seed(seed)
    return SimulatedAnnealing.run(
        problem,
        t_init=t0,
        alpha=alpha,
        t_min=t_min,
        callback=callback,
    )


def parse_int_list(raw_text):
    values = []
    for item in raw_text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("Debes ingresar al menos un valor de n.")
    return unique_values


def parse_float_list(raw_text):
    values = []
    for item in raw_text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("Debes ingresar al menos un valor.")
    return unique_values


def plot_current_state(state, problem, placeholder, problem_name):
    fig, ax = plt.subplots(figsize=(5, 4))

    if problem_name == "TSP (Viajero)":
        coords = problem.coords[state]
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            "o-",
            markersize=5,
            linewidth=1,
            color="#1f77b4",
        )
        ax.plot(
            [coords[-1, 0], coords[0, 0]],
            [coords[-1, 1], coords[0, 1]],
            "o-",
            color="#d62728",
            linewidth=2,
        )
        ax.set_title(f"Ruta actual (Dist: {problem.fitness(state):.2f})")
    else:
        board = np.zeros((problem.n, problem.n))
        for col, row in enumerate(state):
            board[row, col] = 1
        ax.imshow(board, cmap="Blues")
        ax.set_xticks(range(problem.n))
        ax.set_yticks(range(problem.n))
        ax.grid(True, color="gray", linestyle="-", linewidth=0.5)
        ax.set_title(f"Tablero (Conflictos: {problem.fitness(state)})")

    placeholder.pyplot(fig)
    plt.close(fig)


def run_single_experiment(problem_name, n_size, t0, alpha, t_min, seed):
    problem = create_problem(problem_name, n_size, seed=seed)

    hc_seed = seed + 101
    sa_seed = seed + 101

    start_hc = time.perf_counter()
    best_hc, fit_hc, hist_hc = run_hill_climbing(problem, hc_seed)
    time_hc = time.perf_counter() - start_hc

    start_sa = time.perf_counter()
    best_sa, fit_sa, hist_sa = run_simulated_annealing(
        problem,
        t0,
        alpha,
        t_min,
        sa_seed,
    )
    time_sa = time.perf_counter() - start_sa

    return {
        "problem": problem,
        "best_hc": best_hc,
        "fit_hc": fit_hc,
        "hist_hc": hist_hc,
        "time_hc": time_hc,
        "best_sa": best_sa,
        "fit_sa": fit_sa,
        "hist_sa": hist_sa,
        "time_sa": time_sa,
    }


def benchmark_algorithms(problem_name, n_values, trials, t0, alpha, t_min, base_seed):
    records = []

    for n_size in n_values:
        for trial in range(trials):
            problem_seed = base_seed + n_size * 1000 + trial
            problem = create_problem(problem_name, n_size, seed=problem_seed)
            hc_seed = problem_seed + 101
            sa_seed = problem_seed + 101

            start_hc = time.perf_counter()
            _, fit_hc, _ = run_hill_climbing(problem, hc_seed)
            time_hc = time.perf_counter() - start_hc

            start_sa = time.perf_counter()
            _, fit_sa, _ = run_simulated_annealing(problem, t0, alpha, t_min, sa_seed)
            time_sa = time.perf_counter() - start_sa

            if fit_sa < fit_hc:
                winner = "Simulated Annealing"
            elif fit_hc < fit_sa:
                winner = "Hill Climbing"
            else:
                winner = "Empate"

            records.append(
                {
                    "Problema": problem_name,
                    "n": n_size,
                    "Trial": trial + 1,
                    "HC Fitness": fit_hc,
                    "HC Tiempo (s)": time_hc,
                    "SA Fitness": fit_sa,
                    "SA Tiempo (s)": time_sa,
                    "Ganador": winner,
                }
            )

    return pd.DataFrame(records)


def analyze_sa_hyperparameters(problem_name, n_size, trials, t0_values, alpha_values, tmin_values, base_seed):
    baseline_rows = []
    for trial in range(trials):
        problem_seed = base_seed + n_size * 1000 + trial
        problem = create_problem(problem_name, n_size, seed=problem_seed)
        hc_seed = problem_seed + 101

        start_hc = time.perf_counter()
        _, fit_hc, _ = run_hill_climbing(problem, hc_seed)
        time_hc = time.perf_counter() - start_hc

        baseline_rows.append(
            {
                "Trial": trial + 1,
                "HC Fitness": fit_hc,
                "HC Tiempo (s)": time_hc,
                "Problem Seed": problem_seed,
            }
        )

    baseline_df = pd.DataFrame(baseline_rows)

    result_rows = []
    for t0, alpha, t_min in itertools.product(t0_values, alpha_values, tmin_values):
        for trial in range(trials):
            problem_seed = int(baseline_df.iloc[trial]["Problem Seed"])
            problem = create_problem(problem_name, n_size, seed=problem_seed)
            sa_seed = problem_seed + 101

            start_sa = time.perf_counter()
            _, fit_sa, hist_sa = run_simulated_annealing(problem, t0, alpha, t_min, sa_seed)
            time_sa = time.perf_counter() - start_sa

            hc_fit = float(baseline_df.iloc[trial]["HC Fitness"])
            hc_time = float(baseline_df.iloc[trial]["HC Tiempo (s)"])

            result_rows.append(
                {
                    "T0": t0,
                    "Alpha": alpha,
                    "Tmin": t_min,
                    "Trial": trial + 1,
                    "SA Fitness": fit_sa,
                    "SA Tiempo (s)": time_sa,
                    "Pasos SA": len(hist_sa),
                    "HC Fitness": hc_fit,
                    "HC Tiempo (s)": hc_time,
                    "SA mejora a HC": int(fit_sa < hc_fit),
                    "SA empata a HC": int(fit_sa == hc_fit),
                    "SA resuelve": int(fit_sa == 0) if problem_name == "N-Reinas" else np.nan,
                }
            )

    detailed_df = pd.DataFrame(result_rows)
    summary_df = (
        detailed_df.groupby(["T0", "Alpha", "Tmin"], as_index=False)
        .agg(
            Fitness_promedio=("SA Fitness", "mean"),
            Mejor_fitness=("SA Fitness", "min"),
            Tiempo_promedio_s=("SA Tiempo (s)", "mean"),
            Pasos_promedio=("Pasos SA", "mean"),
            Veces_mejor_que_HC=("SA mejora a HC", "sum"),
            Veces_empate_con_HC=("SA empata a HC", "sum"),
        )
        .sort_values(
            by=["Fitness_promedio", "Mejor_fitness", "Tiempo_promedio_s"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )

    if problem_name == "N-Reinas":
        solved_rate = (
            detailed_df.groupby(["T0", "Alpha", "Tmin"], as_index=False)["SA resuelve"]
            .mean()
            .rename(columns={"SA resuelve": "Tasa_solucion"})
        )
        summary_df = summary_df.merge(solved_rate, on=["T0", "Alpha", "Tmin"], how="left")

    return baseline_df, detailed_df, summary_df


def recommendation_text(best_row, problem_name, trials):
    lines = [
        f"Mejor configuracion observada para {problem_name}: "
        f"T0={best_row['T0']}, alpha={best_row['Alpha']}, Tmin={best_row['Tmin']}.",
        f"En {trials} corridas obtuvo fitness promedio {best_row['Fitness_promedio']:.4f} "
        f"y tiempo promedio {best_row['Tiempo_promedio_s']:.4f} s.",
    ]

    if "Veces_mejor_que_HC" in best_row:
        lines.append(
            f"Supero a Hill Climbing en {int(best_row['Veces_mejor_que_HC'])} de {trials} corridas."
        )

    lines.extend(
        [
            "Interpretacion rapida:",
            "T0 alto permite explorar mas y escapar de maximos locales al inicio.",
            "Alpha cercano a 1 enfria mas lento; suele mejorar la calidad, pero aumenta el tiempo.",
            "Tmin pequeno da mas iteraciones finales y refina la solucion, aunque encarece la ejecucion.",
        ]
    )

    return lines


st.title("Comparativa de Algoritmos: Hill Climbing vs Simulated Annealing")
st.sidebar.header("Configuracion general")

problem_type = st.sidebar.selectbox("Selecciona el problema", ["TSP (Viajero)", "N-Reinas"])
seed_value = st.sidebar.number_input("Semilla base", min_value=1, value=42, step=1)

st.sidebar.header("Parametros base de SA")
t0 = st.sidebar.number_input("Temperatura inicial (T0)", min_value=0.01, value=100.0)
alpha = st.sidebar.slider("Enfriamiento (alpha)", 0.80, 0.99, 0.95)
t_min = st.sidebar.number_input("Temperatura final", min_value=0.0001, value=0.1, format="%.4f")

tab_single, tab_benchmark, tab_hyper = st.tabs(
    ["Ejecucion individual", "Benchmark por n", "Analisis de hiperparametros"]
)

with tab_single:
    st.subheader("Comparacion directa en una sola instancia")
    n_size = st.slider("Tamano del problema (n)", 4, 60, 15, key="single_n")

    if st.button("Iniciar optimizacion", key="single_run"):
        problem = create_problem(problem_type, n_size, seed=seed_value)

        col_vis1, col_vis2 = st.columns(2)

        with col_vis1:
            st.subheader("Visualizacion en tiempo real")
            viz_placeholder = st.empty()

        with col_vis2:
            st.subheader("Metricas de desempeno")
            metrics_placeholder = st.empty()

        def update_ui(current_state, current_fit):
            _ = current_fit
            plot_current_state(current_state, problem, viz_placeholder, problem_type)

        st.toast("Ejecutando Hill Climbing...")
        start_hc = time.perf_counter()
        best_hc, fit_hc, hist_hc = run_hill_climbing(problem, seed_value + 101, callback=update_ui)
        time_hc = time.perf_counter() - start_hc

        st.toast("Ejecutando Simulated Annealing...")
        start_sa = time.perf_counter()
        best_sa, fit_sa, hist_sa = run_simulated_annealing(
            problem,
            t0,
            alpha,
            t_min,
            seed_value + 101,
            callback=update_ui,
        )
        time_sa = time.perf_counter() - start_sa

        with metrics_placeholder.container():
            st.write("### Resultados finales")
            st.table(
                pd.DataFrame(
                    {
                        "Algoritmo": ["Hill Climbing", "Simulated Annealing"],
                        "Mejor Fitness": [fit_hc, fit_sa],
                        "Tiempo (s)": [f"{time_hc:.4f}", f"{time_sa:.4f}"],
                    }
                )
            )

            if fit_sa < fit_hc:
                st.success(
                    f"Simulated Annealing logro una mejor solucion por {fit_hc - fit_sa:.2f} unidades."
                )
            elif fit_hc < fit_sa:
                st.warning(
                    "Hill Climbing fue superior; probablemente los parametros de SA enfrien demasiado rapido."
                )
            else:
                st.info("Ambos algoritmos llegaron al mismo resultado.")

        plot_current_state(best_sa, problem, viz_placeholder, problem_type)

        st.divider()
        st.subheader("Analisis de convergencia")
        fig_conv, ax_conv = plt.subplots(figsize=(10, 4))
        ax_conv.plot(
            hist_hc,
            label=f"Hill Climbing (Final: {fit_hc:.2f})",
            color="#d62728",
            alpha=0.8,
        )
        ax_conv.plot(
            hist_sa,
            label=f"Simulated Annealing (Final: {fit_sa:.2f})",
            color="#1f77b4",
            alpha=0.8,
        )
        ax_conv.set_xlabel("Iteraciones / Pasos")
        ax_conv.set_ylabel("Fitness (menor es mejor)")
        ax_conv.legend()
        st.pyplot(fig_conv)
    else:
        st.info("Ejecuta una instancia para ver la solucion paso a paso.")

with tab_benchmark:
    st.subheader("Comparativa por diferentes tamanos de n")
    st.caption(
        "Usa varios tamanos de n y multiples corridas para medir fitness, tiempo y ganador."
    )

    n_values_text = st.text_input("Valores de n separados por comas", value="4, 8, 12, 16, 20")
    trial_count = st.slider("Numero de corridas por cada n", 1, 10, 3)

    if st.button("Ejecutar benchmark", key="benchmark_run"):
        try:
            n_values = parse_int_list(n_values_text)
        except ValueError as exc:
            st.error(str(exc))
        else:
            benchmark_df = benchmark_algorithms(
                problem_type,
                n_values,
                trial_count,
                t0,
                alpha,
                t_min,
                seed_value,
            )

            st.write("### Resultados detallados")
            st.dataframe(benchmark_df, use_container_width=True)

            summary_rows = []
            for n_size in n_values:
                subset = benchmark_df[benchmark_df["n"] == n_size]
                summary_rows.extend(
                    [
                        {
                            "n": n_size,
                            "Algoritmo": "Hill Climbing",
                            "Fitness promedio": subset["HC Fitness"].mean(),
                            "Mejor fitness": subset["HC Fitness"].min(),
                            "Tiempo promedio (s)": subset["HC Tiempo (s)"].mean(),
                            "Victorias": int((subset["Ganador"] == "Hill Climbing").sum()),
                        },
                        {
                            "n": n_size,
                            "Algoritmo": "Simulated Annealing",
                            "Fitness promedio": subset["SA Fitness"].mean(),
                            "Mejor fitness": subset["SA Fitness"].min(),
                            "Tiempo promedio (s)": subset["SA Tiempo (s)"].mean(),
                            "Victorias": int((subset["Ganador"] == "Simulated Annealing").sum()),
                        },
                    ]
                )

            summary_df = pd.DataFrame(summary_rows)
            st.write("### Resumen agregado")
            st.dataframe(summary_df, use_container_width=True)

            fig_fit, ax_fit = plt.subplots(figsize=(10, 4))
            for algo_name, fitness_col in [
                ("Hill Climbing", "HC Fitness"),
                ("Simulated Annealing", "SA Fitness"),
            ]:
                grouped = benchmark_df.groupby("n")[fitness_col].mean()
                ax_fit.plot(grouped.index, grouped.values, marker="o", label=algo_name)
            ax_fit.set_xlabel("n")
            ax_fit.set_ylabel("Fitness promedio")
            ax_fit.set_title("Calidad de solucion por tamano de problema")
            ax_fit.legend()
            st.pyplot(fig_fit)

            fig_time, ax_time = plt.subplots(figsize=(10, 4))
            for algo_name, time_col in [
                ("Hill Climbing", "HC Tiempo (s)"),
                ("Simulated Annealing", "SA Tiempo (s)"),
            ]:
                grouped = benchmark_df.groupby("n")[time_col].mean()
                ax_time.plot(grouped.index, grouped.values, marker="o", label=algo_name)
            ax_time.set_xlabel("n")
            ax_time.set_ylabel("Tiempo promedio (s)")
            ax_time.set_title("Costo computacional por tamano de problema")
            ax_time.legend()
            st.pyplot(fig_time)

            winner_counts = (
                benchmark_df.groupby(["n", "Ganador"]).size().unstack(fill_value=0).sort_index()
            )
            st.write("### Ganador por n")
            st.bar_chart(winner_counts)

with tab_hyper:
    st.subheader("Analisis de hiperparametros de Simulated Annealing")
    st.caption(
        "Se evalua una grilla de T0, alpha y Tmin para medir calidad, tiempo y ventaja frente a Hill Climbing."
    )

    hyper_n = st.slider("Tamano n para el barrido de parametros", 4, 60, 15, key="hyper_n")
    hyper_trials = st.slider("Corridas por configuracion", 1, 10, 3, key="hyper_trials")
    t0_grid_text = st.text_input("Valores de T0", value="10, 50, 100, 200")
    alpha_grid_text = st.text_input("Valores de alpha", value="0.85, 0.90, 0.95, 0.99")
    tmin_grid_text = st.text_input("Valores de Tmin", value="1, 0.1, 0.01")

    if st.button("Analizar hiperparametros", key="hyper_run"):
        try:
            t0_values = parse_float_list(t0_grid_text)
            alpha_values = parse_float_list(alpha_grid_text)
            tmin_values = parse_float_list(tmin_grid_text)
        except ValueError as exc:
            st.error(str(exc))
        else:
            baseline_df, hyper_df, hyper_summary = analyze_sa_hyperparameters(
                problem_type,
                hyper_n,
                hyper_trials,
                t0_values,
                alpha_values,
                tmin_values,
                seed_value,
            )

            st.write("### Baseline de Hill Climbing")
            st.dataframe(baseline_df.drop(columns=["Problem Seed"]), use_container_width=True)

            st.write("### Resultados agregados de SA")
            st.dataframe(hyper_summary, use_container_width=True)

            st.write("### Top 10 configuraciones")
            st.dataframe(hyper_summary.head(10), use_container_width=True)

            best_row = hyper_summary.iloc[0]
            st.write("### Recomendacion")
            for line in recommendation_text(best_row, problem_type, hyper_trials):
                st.write(f"- {line}")

            fig_hyper_fit, ax_hyper_fit = plt.subplots(figsize=(10, 4))
            ax_hyper_fit.plot(
                range(1, len(hyper_summary) + 1),
                hyper_summary["Fitness_promedio"],
                marker="o",
                color="#1f77b4",
            )
            ax_hyper_fit.set_xlabel("Ranking de configuraciones")
            ax_hyper_fit.set_ylabel("Fitness promedio")
            ax_hyper_fit.set_title("Ranking de calidad de SA")
            st.pyplot(fig_hyper_fit)

            fig_hyper_time, ax_hyper_time = plt.subplots(figsize=(10, 4))
            ax_hyper_time.plot(
                range(1, len(hyper_summary) + 1),
                hyper_summary["Tiempo_promedio_s"],
                marker="o",
                color="#d62728",
            )
            ax_hyper_time.set_xlabel("Ranking de configuraciones")
            ax_hyper_time.set_ylabel("Tiempo promedio (s)")
            ax_hyper_time.set_title("Costo temporal por configuracion")
            st.pyplot(fig_hyper_time)

            if len(alpha_values) > 1:
                alpha_effect = hyper_df.groupby("Alpha", as_index=False).agg(
                    Fitness_promedio=("SA Fitness", "mean"),
                    Tiempo_promedio_s=("SA Tiempo (s)", "mean"),
                )
                st.write("### Efecto de alpha")
                st.dataframe(alpha_effect, use_container_width=True)

            if len(t0_values) > 1:
                t0_effect = hyper_df.groupby("T0", as_index=False).agg(
                    Fitness_promedio=("SA Fitness", "mean"),
                    Tiempo_promedio_s=("SA Tiempo (s)", "mean"),
                )
                st.write("### Efecto de T0")
                st.dataframe(t0_effect, use_container_width=True)

            if len(tmin_values) > 1:
                tmin_effect = hyper_df.groupby("Tmin", as_index=False).agg(
                    Fitness_promedio=("SA Fitness", "mean"),
                    Tiempo_promedio_s=("SA Tiempo (s)", "mean"),
                )
                st.write("### Efecto de Tmin")
                st.dataframe(tmin_effect, use_container_width=True)
    else:
        st.info("Lanza el barrido para identificar parametros convenientes de SA.")
