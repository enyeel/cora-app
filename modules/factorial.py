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

    # 🛑 PARCHE ANTI-NaNs: Eliminar columnas constantes (varianza == 0)
    # Las variables sin variación rompen el KMO y dan valores NaN.
    df = df.loc[:, df.apply(pd.Series.nunique) > 1]

    if df.shape[1] < 2:
        raise ValueError("Se necesitan al menos 2 variables numéricas con variación.")

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


# ====================================================================
# 3. NORMALIZATION
# ====================================================================
def normalizar(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# ====================================================================
# 4. SCREE PLOT (PLOTLY)
# ====================================================================
def generar_scree_plot(df):
    fa = FactorAnalyzer()
    fa.fit(df)

    eigenvalues, _ = fa.get_eigenvalues()

    fig = px.line(
        x=list(range(1, len(eigenvalues) + 1)),
        y=eigenvalues,
        markers=True,
        title='Scree Plot (Eigenvalues)',
        labels={'x': 'Number of Factors', 'y': 'Eigenvalues'},
        template='plotly_dark'
    )

    return fig, eigenvalues


# ====================================================================
# 5. FACTOR DIAGRAM
# ====================================================================
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def generar_diagrama_plotly(cargas, umbral=0.4):
    # Group and sort variables to avoid tangled connections
    # Find the dominant factor for each variable (highest loading)
    var_info = []
    for var in cargas.index:
        cargas_var = cargas.loc[var].abs()
        max_factor = cargas_var.idxmax()
        max_val = cargas_var.max()
        factor_idx = cargas.columns.get_loc(max_factor)
        
        var_info.append({
            'var': var,
            'factor_idx': factor_idx,
            'max_val': max_val
        })
        
    # Ordenamos: primero por factor al que pertenecen, luego por la fuerza de la carga
    var_info.sort(key=lambda x: (x['factor_idx'], -x['max_val']))
    variables_ordenadas = [x['var'] for x in var_info]
    var_to_factor_idx = {x['var']: x['factor_idx'] for x in var_info}
    
    factores = [str(c) if "Factor" in str(c) else f"Factor {c+1}" for c in cargas.columns]
    
    # 2. COLORES CHIDOS PARA CADA FACTOR
    paleta = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    def get_color(idx):
        return paleta[idx % len(paleta)]
    
    # Coordenadas
    y_vars = np.linspace(1, 0, len(variables_ordenadas))
    y_facts = np.linspace(1, 0, len(factores))
    
    fig = go.Figure()
    
    # 3. DIBUJAR LÍNEAS (Ahora con color del factor y punteadas si son negativas)
    for i, var in enumerate(variables_ordenadas):
        for j, factor_col in enumerate(cargas.columns):
            peso = cargas.loc[var, factor_col]
            
            if abs(peso) >= umbral:
                color_base = get_color(j)
                grosor = abs(peso) * 6
                estilo_linea = 'solid' if peso > 0 else 'dot' # Punteado si es negativo
                
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[y_vars[i], y_facts[j]],
                    mode='lines',
                    line=dict(width=grosor, color=color_base, dash=estilo_linea),
                    opacity=0.6,
                    hoverinfo='text',
                    text=f"{var} ➔ {factores[j]}<br>Carga: {peso:.3f}"
                ))
                
    # 4. NODOS DE VARIABLES (Con el color de su factor principal)
    colores_vars = [get_color(var_to_factor_idx[var]) for var in variables_ordenadas]
    fig.add_trace(go.Scatter(
        x=[0] * len(variables_ordenadas),
        y=y_vars,
        mode='markers+text',
        marker=dict(size=18, color=colores_vars, symbol='square', line=dict(color='white', width=1)),
        text=variables_ordenadas,
        textposition='middle left',
        hoverinfo='none',
        cliponaxis=False # Magia para que no se corten las etiquetas 🪄
    ))
    
    # 5. NODOS DE FACTORES
    fig.add_trace(go.Scatter(
        x=[1] * len(factores),
        y=y_facts,
        mode='markers+text',
        marker=dict(size=40, color=[get_color(i) for i in range(len(factores))], symbol='circle', line=dict(color='white', width=2)),
        text=factores,
        textposition='middle right',
        hoverinfo='none',
        cliponaxis=False # Magia para que no se corten las etiquetas 🪄
    ))
    
    # 6. TUNEAR LA GRÁFICA (Para que ocupe todo el ancho)
    altura_dinamica = max(500, len(variables_ordenadas) * 40) 
    
    fig.update_layout(
        title="Diagrama de Senderos Estructurado",
        template="plotly_dark",
        showlegend=False,
        height=altura_dinamica,
        margin=dict(l=250, r=150, t=50, b=50), # Márgenes grandes para los textos
        # Rango más ajustado al [0, 1] para que la gráfica no sea un fideo enmedio
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.1, 1.1]), 
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

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

    return fig


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
    


def ordenar_matriz_cargas(df_cargas):
    """
    Ordena un DataFrame de cargas factoriales agrupando las variables 
    por su factor principal (donde tienen la carga absoluta más alta).
    """
    var_info = []
    for var in df_cargas.index:
        # Tomamos valores absolutos para encontrar el factor más fuerte
        cargas_var = df_cargas.loc[var].abs()
        max_factor = cargas_var.idxmax()
        max_val = cargas_var.max()
        factor_idx = df_cargas.columns.get_loc(max_factor)
        
        var_info.append({
            'var': var,
            'factor_idx': factor_idx,
            'max_val': max_val
        })
        
    # Ordenamos: primero por el factor al que pertenecen, luego por la fuerza de la carga
    var_info.sort(key=lambda x: (x['factor_idx'], -x['max_val']))
    variables_ordenadas = [x['var'] for x in var_info]
    
    # Devolvemos el DataFrame con el nuevo orden de filas
    return df_cargas.loc[variables_ordenadas]