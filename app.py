"""
Recurso interactivo - Situación 3A: Rendimiento de una aplicación
Cálculo diferencial 2026-II - Ingeniería de Software

Modelos (x en miles de usuarios, t en semanas):
  T(x)  = (5x + 40)/(x - 4)            versión original  [s]
  T*(x) = (x^2 - 16)/(x^2 - 4x)        nueva versión     [s]  (= (x+4)/x para x != 4)
  x(t)  = 20 / (1 + 9 e^(-0.4 t))      usuarios          [miles]
  C(n)  = 120 n + (4800 + 200k)/n      costo mensual     [miles de pesos]
"""
import numpy as np
import sympy as sp
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Situación 3A - Rendimiento de una aplicación", layout="wide")

# ----------------------------------------------------------------- Parámetros
st.sidebar.header("Parámetros")
k = st.sidebar.number_input("k (3)", min_value=0, max_value=100, value=1, step=1,
                            )
CAP = st.sidebar.number_input("Capacidad actual (miles de usuarios)", 5.0, 100.0, 20.0, 1.0)
A = st.sidebar.number_input("Constante a de x(t) (el 9 de 1 + 9e^(-rt))", 1.0, 50.0, 9.0, 0.5)
R = st.sidebar.slider("Tasa de crecimiento r (1/semana)", 0.1, 1.0, 0.4, 0.05)
SERV_COST = st.sidebar.number_input("Costo por servidor (miles de pesos/mes)", 10.0, 1000.0, 120.0, 10.0)
N_MAX = st.sidebar.number_input("Servidores que caben en el centro de datos", 1, 100, 20, 1)
x_sel = st.sidebar.slider("Usuarios a analizar x (miles)", 0.5, float(CAP), 6.0, 0.1)
t_sel = st.sidebar.slider("Semana a analizar t", 0.0, 24.0, 4.0, 0.5)
umbral = st.sidebar.slider("Umbral de alerta (% de la capacidad)", 50, 100, 90, 5)

# ----------------------------------------------------------------- Cálculo simbólico (SymPy)
x, t, n = sp.symbols("x t n", positive=True)
T = (5 * x + 40) / (x - 4)
Ts = (x**2 - 16) / (x**2 - 4 * x)
X = CAP / (1 + A * sp.exp(-R * t))
C = SERV_COST * n + (4800 + 200 * k) / n

dT, dTs, dX, ddX = sp.diff(T, x), sp.diff(Ts, x), sp.diff(X, t), sp.diff(X, t, 2)
dC, ddC = sp.diff(C, n), sp.diff(C, n, 2)
f_T, f_Ts = sp.lambdify(x, T, "numpy"), sp.lambdify(x, Ts, "numpy")
f_dT, f_dTs = sp.lambdify(x, dT, "numpy"), sp.lambdify(x, sp.simplify(dTs), "numpy")
f_X, f_dX = sp.lambdify(t, X, "numpy"), sp.lambdify(t, dX, "numpy")
f_C = sp.lambdify(n, C, "numpy")

lim_Ts_4 = sp.limit(Ts, x, 4)  # valor que se asigna a T*(4)

# Punto de inflexión de x(t): x''(t)=0  ->  A e^(-R t) = 1
t_infl = float(np.log(A) / R)
x_infl = float(f_X(t_infl))

# Servidores óptimos (entero): se comparan los enteros vecinos del punto crítico
n_crit = float(np.sqrt((4800 + 200 * k) / SERV_COST))
n_vals = np.arange(1, int(N_MAX) + 1)
costos = f_C(n_vals.astype(float))
n_opt = int(n_vals[np.argmin(costos)])
c_opt = float(costos.min())

# Semana en que se cruza el umbral de alerta
thr = CAP * umbral / 100
t_thr = float(-np.log((CAP / thr - 1) / A) / R) if thr < CAP else float("inf")

# ----------------------------------------------------------------- Encabezado
st.title("Situación 3A - Rendimiento de una aplicación")
st.caption(f"Grupo k = {k}  |  x en miles de usuarios, T en segundos, t en semanas, costos en miles de pesos/mes")

tab1, tab2, tab3, tab4 = st.tabs(["1. Versiones (T y T*)", "2. Usuarios x(t)", "3. Servidores", "4. Zona de decisión"])

# ----------------------------------------------------------------- Pestaña 1
with tab1:
    st.subheader("Dominio")
    st.markdown(
        f"- **T(x):** x ∈ (4, {CAP:g}] miles de usuarios. Se excluye x = 4 (asíntota vertical) y, para x < 4, T sería negativo (sin sentido físico). "
        f"El extremo superior es la capacidad de la infraestructura.\n"
        f"- **T\\*(x):** x ∈ (0, {CAP:g}]. Hay una discontinuidad removible en x = 4; con el límite se define "
        f"**T\\*(4) = {lim_Ts_4}** s."
    )
    xs_T = np.linspace(4.05, CAP, 600)
    xs_Ts = np.linspace(0.3, CAP, 600)

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=xs_T, y=f_T(xs_T), name="T(x) original")
    fig.add_scatter(x=xs_Ts, y=f_Ts(xs_Ts), name="T*(x) nueva")
    fig.add_vline(x=x_sel, line_dash="dot")
    fig.update_layout(title="Tiempo de respuesta", xaxis_title="Usuarios (miles)",
                      yaxis_title="Tiempo de respuesta (s)", yaxis_range=[0, 60])
    c1.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_scatter(x=xs_T, y=f_dT(xs_T), name="T'(x) original")
    fig.add_scatter(x=xs_Ts, y=f_dTs(xs_Ts), name="T*'(x) nueva")
    fig.add_vline(x=x_sel, line_dash="dot")
    fig.update_layout(title="Ritmo de cambio del tiempo", xaxis_title="Usuarios (miles)",
                      yaxis_title="s por cada mil usuarios", yaxis_range=[-30, 0])
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Indicadores con x = {x_sel:g} mil usuarios")
    m = st.columns(4)
    if abs(x_sel - 4) < 1e-9:
        m[0].metric("T(x)", "no definido")
        m[1].metric("T'(x)", "no existe")
        m[2].metric("T*(x)", f"{float(lim_Ts_4):.3f} s (por límite)")
        m[3].metric("T*'(x)", f"{float(f_dTs(4.0)):.4f} s/mil")
    else:
        m[0].metric("T(x)", f"{float(f_T(x_sel)):.3f} s" if x_sel > 4 else "fuera del dominio")
        m[1].metric("T'(x)", f"{float(f_dT(x_sel)):.4f} s/mil" if x_sel > 4 else "fuera del dominio")
        m[2].metric("T*(x)", f"{float(f_Ts(x_sel)):.3f} s")
        m[3].metric("T*'(x)", f"{float(f_dTs(x_sel)):.4f} s/mil")
    st.info("Derivadas calculadas con SymPy:  T'(x) = −60/(x−4)²   y   T*'(x) = −4/x².  "
            "Ambas son negativas: con más usuarios el tiempo baja; T* es más estable porque su ritmo es mucho menor.")

# ----------------------------------------------------------------- Pestaña 2
with tab2:
    ts = np.linspace(0, 24, 400)
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=ts, y=f_X(ts), name="x(t)")
    fig.add_hline(y=CAP, line_dash="dash", annotation_text="Capacidad / asíntota")
    fig.add_hline(y=thr, line_dash="dot", annotation_text=f"Umbral {umbral}%")
    fig.add_scatter(x=[t_infl], y=[x_infl], mode="markers", marker_size=11, name="Punto de inflexión")
    fig.update_layout(title="Usuarios en el tiempo", xaxis_title="Semanas", yaxis_title="Usuarios (miles)")
    c1.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_scatter(x=ts, y=f_dX(ts), name="x'(t)")
    fig.add_scatter(x=[t_infl], y=[float(f_dX(t_infl))], mode="markers", marker_size=11, name="Máximo ritmo")
    fig.update_layout(title="Ritmo de crecimiento de usuarios", xaxis_title="Semanas",
                      yaxis_title="Miles de usuarios por semana")
    c2.plotly_chart(fig, use_container_width=True)

    m = st.columns(4)
    m[0].metric("Semana de mayor crecimiento", f"{t_infl:.2f}")
    m[1].metric("Usuarios en ese momento", f"{x_infl:.2f} mil")
    m[2].metric("Ritmo máximo x'(t)", f"{float(f_dX(t_infl)):.2f} mil/sem")
    m[3].metric("Límite (t→∞)", f"{CAP:g} mil")

    xt = float(f_X(t_sel))
    rate_t = float(f_dTs(xt) * f_dX(t_sel))  # regla de la cadena
    st.subheader(f"Semana t = {t_sel:g}")
    m = st.columns(4)
    m[0].metric("Usuarios x(t)", f"{xt:.3f} mil")
    m[1].metric("x'(t)", f"{float(f_dX(t_sel)):.3f} mil/sem")
    m[2].metric("T*(x(t))", f"{float(f_Ts(xt)):.3f} s")
    m[3].metric("dT*/dt (regla de la cadena)", f"{rate_t:.4f} s/sem")
    st.info("x'(t) alcanza su máximo cuando x'' = 0, es decir cuando A·e^(−rt) = 1 → t = ln(A)/r, y ahí x = capacidad/2. "
            "El límite de x(t) cuando t→∞ coincide con la capacidad de la infraestructura (20 mil usuarios).")

# ----------------------------------------------------------------- Pestaña 3
with tab3:
    st.markdown(f"**C(n) = {SERV_COST:g}·n + (4800 + 200·{k})/n**  con n ∈ {{1, …, {int(N_MAX)}}} (n entero, límite del centro de datos)")
    fig = go.Figure()
    colores = ["crimson" if v == n_opt else "steelblue" for v in n_vals]
    fig.add_bar(x=n_vals, y=costos, marker_color=colores, name="C(n)")
    nn = np.linspace(0.5, int(N_MAX), 400)
    fig.add_scatter(x=nn, y=f_C(nn), name="C(n) continuo")
    fig.update_layout(title="Costo mensual según servidores", xaxis_title="Servidores n",
                      yaxis_title="Miles de pesos por mes")
    st.plotly_chart(fig, use_container_width=True)

    m = st.columns(4)
    m[0].metric("Punto crítico n* = √((4800+200k)/costo)", f"{n_crit:.2f}")
    m[1].metric("n óptimo (entero)", f"{n_opt}")
    m[2].metric("Costo mínimo", f"{c_opt:,.1f} mil/mes")
    m[3].metric("C''(n) en n*", f"{float(ddC.subs(n, n_crit)):.3f} (>0 → mínimo)")
    vec = [v for v in (int(np.floor(n_crit)), int(np.ceil(n_crit))) if 1 <= v <= N_MAX]
    #st.write("Comparación de enteros vecinos:", {v: round(float(f_C(float(v))), 2) for v in sorted(set(vec))})

# ----------------------------------------------------------------- Pestaña 4
with tab4:
    st.subheader("Zona de decisión")
    xs = np.linspace(4.05, CAP, 400)
    t_menor = bool(np.all(f_Ts(xs) < f_T(xs)))
    version = "Nueva versión (T*)" if t_menor else "Revisar: la nueva versión no siempre es mejor"
    m = st.columns(3)
    m[0].metric("Versión a publicar", version)
    m[1].metric("Servidores a contratar", f"{n_opt}")
    m[2].metric("Reforzar infraestructura en la semana", f"{t_infl:.1f}")
    st.markdown(
        f"- La nueva versión tiene menor tiempo **y** menor ritmo de variación en todo el dominio de T (x > 4): "
        f"{'se cumple con los datos actuales.' if t_menor else 'no se cumple con los datos actuales.'}\n"
        f"- Con {n_opt} servidores el costo mensual es el mínimo: **{c_opt:,.1f} mil pesos**.\n"
        f"- El crecimiento de usuarios es más rápido en la semana **{t_infl:.1f}** ({x_infl:.1f} mil usuarios): reforzar antes de esa semana.\n"
        f"- Se supera el umbral de {umbral}% de la capacidad ({thr:.1f} mil usuarios) en la semana "
        f"**{t_thr:.1f}**." if np.isfinite(t_thr) else "- El umbral elegido no se alcanza."
    )
    if x_infl >= CAP:
        st.warning("El punto de inflexión supera la capacidad actual.")

# ----------------------------------------------------------------- Código visible

