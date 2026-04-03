import pandas as pd
import numpy as np

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_kmo,
    calculate_bartlett_sphericity
)

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.stats import norm

import plotly.express as px
import plotly.graph_objects as go


# -------------------------------
# 🔹 1. LIMPIEZA UNIVERSAL
# -------------------------------
def limpiar_datos(df):
    df = df.select_dtypes(include=['number'])
    df = df.dropna(axis=1, how='all')
    df = df.fillna(df.mean())

    if df.shape[1] < 2:
        raise ValueError("Se necesitan al menos 2 variables numéricas")

    return df


# -------------------------------
# 🔹 2. ELIMINAR MULTICOLINEALIDAD
# -------------------------------
def eliminar_multicolinealidad(df, umbral=0.95):
    corr = df.corr().abs()
    eliminar = set()

    for i in range(len(corr.columns)):
        for j in range(i):
            if corr.iloc[i, j] > umbral:
                eliminar.add(corr.columns[i])

    return df.drop(columns=eliminar)


# -------------------------------
# 🔹 3. NORMALIZAR
# -------------------------------
def normalizar(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# -------------------------------
# 🔹 4. SCREE PLOT (PLOTLY)
# -------------------------------
def generar_scree_plot(df):
    fa = FactorAnalyzer()
    fa.fit(df)

    eigenvalues, _ = fa.get_eigenvalues()

    fig = px.line(
        x=list(range(1, len(eigenvalues) + 1)),
        y=eigenvalues,
        markers=True,
        title='Scree Plot (Valores Propios)',
        labels={'x': 'Número de Factores', 'y': 'Eigenvalues'},
        template='plotly_dark'
    )

    return fig.to_json(), eigenvalues


# -------------------------------
# 🔹 5. DIAGRAMA DE FACTORES
# -------------------------------
def generar_diagrama_plotly(cargas, umbral=0.4):
    edges_x = []
    edges_y = []

    for var in cargas.index:
        for i, val in enumerate(cargas.loc[var]):
            if abs(val) >= umbral:
                edges_x.extend([0, 1, None])
                edges_y.extend([i, cargas.index.get_loc(var), None])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edges_x,
        y=edges_y,
        mode='lines',
        line=dict(width=2)
    ))

    fig.update_layout(
        title="Diagrama de Factores",
        template="plotly_dark",
        showlegend=False
    )

    return fig.to_json()


# -------------------------------
# 🔹 6. HEATMAP DE CARGAS
# -------------------------------
def heatmap_cargas(cargas):
    fig = px.imshow(
        cargas,
        text_auto=True,
        color_continuous_scale='RdBu',
        title="Cargas Factoriales",
        template="plotly_dark"
    )

    return fig.to_json()


# -------------------------------
# 🔹 7. BOOTSTRAP (INFERENCIA)
# -------------------------------
def bootstrap(df, n_factores, n_bootstrap=50):
    fa = FactorAnalyzer(n_factors=n_factores, rotation='varimax')
    fa.fit(df)
    original = fa.loadings_

    muestras = []

    for _ in range(n_bootstrap):
        sample = resample(df)
        fa.fit(sample)
        muestras.append(fa.loadings_)

    muestras = np.array(muestras)

    ee = muestras.std(axis=0)
    z = original / ee
    p = 2 * (1 - norm.cdf(np.abs(z)))

    return {
        "cargas": original.tolist(),
        "error": ee.tolist(),
        "z": z.tolist(),
        "p": p.tolist()
    }


# -------------------------------
# 🔹 8. FUNCIÓN PRINCIPAL
# -------------------------------
def hacer_calculos(df, modo="auto", n_factores=None):
    try:
        df = limpiar_datos(df)
        df = eliminar_multicolinealidad(df)
        df = normalizar(df)

        kmo = calculate_kmo(df)[1]
        chi2, p = calculate_bartlett_sphericity(df)

        # Scree Plot
        scree_json, eigenvalues = generar_scree_plot(df)

        # Selección automática
        if modo == "auto":
            n_factores = max(sum(eigenvalues > 1), 1)

        if not n_factores:
            return {"status": "error", "mensaje": "Debes especificar n_factores"}

        # Análisis factorial
        fa = FactorAnalyzer(n_factors=n_factores, rotation='varimax', method='minres')
        fa.fit(df)

        cargas = pd.DataFrame(fa.loadings_, index=df.columns)
        factores = pd.DataFrame(fa.transform(df))

        # Visualizaciones
        diagrama_json = generar_diagrama_plotly(cargas)
        heatmap_json = heatmap_cargas(cargas)

        # Inferencia
        inferencia = bootstrap(df, n_factores)

        return {
            "status": "ok",
            "kmo": float(kmo),
            "bartlett_p": float(p),
            "n_factores": int(n_factores),

            "columnas": list(df.columns),

            "cargas": cargas.to_dict(),
            "factores": factores.to_dict(),

            "scree_plot": scree_json,
            "diagrama": diagrama_json,
            "heatmap": heatmap_json,

            "inferencia": inferencia
        }

    except Exception as e:
        return {
            "status": "error",
            "mensaje": str(e)
        }