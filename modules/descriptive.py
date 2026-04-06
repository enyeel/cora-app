"""descriptive.py
Backend de análisis descriptivo 100% migrado a Plotly.
Todas las funciones devuelven DataFrames de pandas o Figuras de Plotly.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kstest, skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go

# ------------------------- frequency / categorical ----------------------
def categorical_frequency_table(data: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    col_data = data[column].dropna()
    freq = col_data.value_counts()
    
    #  ESTRATEGIA TOP N: Agrupar la cola larga en "Otros" 
    if len(freq) > top_n:
        top_freq = freq.iloc[:top_n]
        otros_freq = pd.Series([freq.iloc[top_n:].sum()], index=['Otros'])
        freq = pd.concat([top_freq, otros_freq])
        
    rel_freq = freq / freq.sum()
    cum_freq = freq.cumsum()
    
    table = pd.DataFrame({
        "Category": freq.index.astype(str), # Convertir a string para evitar errores con listas
        "Frequency": freq.values,
        "Relative Frequency": rel_freq.values,
        "Cumulative Frequency": cum_freq.values
    })
    return table

def plot_categorical_bar(table: pd.DataFrame, column: str):
    fig = px.bar(table, x="Category", y="Frequency", text="Frequency",
                 title=f"Gráfico de Barras - {column}", color="Category")
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    return fig

# ------------------------- numeric summaries ---------------------------
def central_tendency(data: pd.DataFrame, column: str) -> pd.DataFrame:
    col_data = data[column].dropna()
    mean = np.mean(col_data)
    median = np.median(col_data)
    # Scipy mode update for newer versions
    mode_result = stats.mode(col_data, keepdims=True)
    mode = mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan
    
    return pd.DataFrame({"Measure": ["Mean", "Median", "Mode"], "Value": [mean, median, mode]})

def dispersion_measures(data: pd.DataFrame, column: str) -> pd.DataFrame:
    data_col = data[column].dropna()
    range_val = np.max(data_col) - np.min(data_col)
    variance = np.var(data_col, ddof=1)
    std_dev = np.std(data_col, ddof=1)
    mean = np.mean(data_col)
    coef_var = (std_dev / mean) * 100 if mean != 0 else None
    
    return pd.DataFrame({
        "Measure": ["Range", "Variance", "Standard Deviation", "Coef of Variation (%)"],
        "Value": [range_val, variance, std_dev, coef_var]
    })

# ------------------------- frequency tables ----------------------------
def frequency_table(data: pd.DataFrame, column: str, bins: int = None) -> pd.DataFrame:
    col_data = data[column].dropna()
    if col_data.empty:
        raise ValueError(f"La columna '{column}' no tiene datos válidos.")
    
    if bins is None:
        n = len(col_data)
        bins = int(1 + 3.322 * np.log10(n)) if n > 0 else 1
        
    intervals = pd.cut(col_data, bins=bins)
    freq = intervals.value_counts().sort_index()
    rel_freq = freq / freq.sum()
    cum_freq = freq.cumsum()
    cum_rel_freq = rel_freq.cumsum()
    class_marks = [((iv.left + iv.right) / 2) for iv in freq.index]
    
    table = pd.DataFrame({
        "Interval": freq.index.astype(str),
        "Class Mark": class_marks,
        "Frequency": freq.values,
        "Relative Frequency": rel_freq.values,
        "Cumulative Frequency": cum_freq.values,
        "Cumulative Relative Frequency": cum_rel_freq.values
    })
    return table

# ------------------------------ plots ----------------------------------
def histogram_from_table(table: pd.DataFrame, column: str):
    fig = px.bar(table, x="Interval", y="Frequency", text="Frequency",
                 title=f"Histograma - {column}")
    fig.update_traces(marker_color='royalblue', marker_line_color='black', marker_line_width=1, opacity=0.8)
    fig.update_layout(xaxis_tickangle=-45, bargap=0)
    return fig

def frequency_polygon(table: pd.DataFrame, column: str):
    fig = px.line(table, x="Class Mark", y="Frequency", markers=True,
                  title=f"Polígono de Frecuencias - {column}")
    fig.update_traces(line=dict(color='firebrick', width=3), marker=dict(size=8))
    return fig

def boxplot(data: pd.DataFrame, column: str):
    #  LÍMITE DE OUTLIERS VISUALES 
    # Si hay demasiados datos, evitamos que Plotly dibuje cada outlier para no colapsar el DOM
    if len(data[column].dropna()) > 5000:
        fig = px.box(data, y=column, title=f"Boxplot - {column} (Outliers visuales ocultos)", points=False)
    else:
        fig = px.box(data, y=column, title=f"Boxplot - {column}", points="outliers")
        
    fig.update_traces(marker_color='seagreen')
    return fig

def ogive(freq_table: pd.DataFrame, column_name: str):
    fig = px.line(freq_table, x="Class Mark", y="Cumulative Relative Frequency", markers=True,
                  title=f"Ojiva (Frecuencia Acumulada) - {column_name}")
    fig.update_traces(line=dict(color='darkorange', width=3), marker=dict(size=8))
    fig.update_layout(yaxis_tickformat='.0%')
    return fig

def scatter_plot(data: pd.DataFrame, x_column: str, y_column: str):
    # Quitamos nulos solo de las columnas que vamos a usar
    plot_data = data[[x_column, y_column]].dropna()
    
    #  LÍMITE DURO ANTI-OOM 
    if len(plot_data) > 3000:
        plot_data = plot_data.sample(3000, random_state=42)
        titulo = f"Dispersión: {x_column} vs {y_column} (Muestra de 3000 pts)"
    else:
        titulo = f"Dispersión: {x_column} vs {y_column}"

    # Uso de render_mode='webgl' para super rendimiento
    fig = px.scatter(plot_data, x=x_column, y=y_column, title=titulo, render_mode='webgl')
    fig.update_traces(marker=dict(opacity=0.6, size=6))
    return fig

def scatter_matrix(data: pd.DataFrame):
    numeric_data = data.select_dtypes(include=['number']).dropna()
    if numeric_data.shape[1] < 2:
        return None
    
    # Muestreo para no colapsar la app si hay más de 2000 datos
    if len(numeric_data) > 2000:
        numeric_data = numeric_data.sample(2000, random_state=42)
        titulo = "Matriz de Dispersión (Muestra de 2000 puntos)"
    else:
        titulo = "Matriz de Dispersión"

    fig = px.scatter_matrix(numeric_data, title=titulo)
    fig.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.5))
    fig.update_layout(height=800)
    return fig

# ------------------------- interpretation / shape ----------------------
def interpret_shape(skewness: float, kurtosis_val: float) -> pd.DataFrame:
    interpretation = []
    if skewness > 0.5:
        interpretation.append("Distribución asimétrica positiva (cola a la derecha).")
    elif skewness < -0.5:
        interpretation.append("Distribución asimétrica negativa (cola a la izquierda).")
    else:
        interpretation.append("Distribución aproximadamente simétrica.")
        
    if kurtosis_val > 0:
        interpretation.append("Colas pesadas (leptocúrtica), indica más valores extremos.")
    elif kurtosis_val < 0:
        interpretation.append("Colas ligeras (platicúrtica), indica menos valores extremos.")
    else:
        interpretation.append("Colas similares a la normal (mesocúrtica).")
        
    return pd.DataFrame({"Interpretación": interpretation})

# --------------------------- normality --------------------------------
def normality_tests(data: pd.DataFrame, column: str) -> pd.DataFrame:
    col_data = data[column].dropna()
    
    # El test de Shapiro crashea con más de 5000 muestras, hacemos subsample si es necesario
    if len(col_data) > 5000:
        col_data_shapiro = col_data.sample(5000, random_state=42)
    else:
        col_data_shapiro = col_data
        
    shapiro_stat, shapiro_p = shapiro(col_data_shapiro)
    mean, std = np.mean(col_data), np.std(col_data, ddof=1)
    ks_stat, ks_p = kstest(col_data, 'norm', args=(mean, std))
    
    alpha = 0.05
    normal_shapiro = shapiro_p > alpha
    normal_ks = ks_p > alpha
    
    if normal_shapiro and normal_ks:
        conclusion = "Los datos tienen distribución normal (tests paramétricos OK)."
        test_type = "Paramétrico"
    else:
        conclusion = "Los datos NO tienen distribución normal (usar no paramétricos)."
        test_type = "No paramétrico"
        
    return pd.DataFrame({
        "Shapiro-Wilk (p-value)": [shapiro_p],
        "Kolmogorov-Smirnov (p-value)": [ks_p],
        "Normal (Shapiro)": [normal_shapiro],
        "Normal (KS)": [normal_ks],
        "Conclusión": [conclusion],
        "Test Recomendado": [test_type]
    })

# ------------------------- position & shape ----------------------------
def position_measures(data: pd.DataFrame, column: str) -> pd.DataFrame:
    col_data = data[column].dropna()
    quartiles = np.percentile(col_data, [25, 50, 75])
    percentiles = np.percentile(col_data, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    return pd.DataFrame({
        "Measure": ["Q1 (25%)", "Q2 (Median/50%)", "Q3 (75%)", "Deciles (10% to 90%)"],
        "Value": [quartiles[0], quartiles[1], quartiles[2], str(np.round(percentiles, 2).tolist())]
    })

def shape_measures(data: pd.DataFrame, column: str) -> pd.DataFrame:
    skewness_val = skew(data[column].dropna())
    kurt = kurtosis(data[column].dropna())
    return pd.DataFrame({"Measure": ["Skewness (Asimetría)", "Kurtosis (Curtosis)"], "Value": [skewness_val, kurt]})

# ------------------------- correlation functions ----------------------
def correlation_matrix(data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    numeric_data = data.select_dtypes(include=['number']).dropna()
    if numeric_data.shape[1] < 2:
        return pd.DataFrame()
    return numeric_data.corr(method=method)

def correlation_heatmap(data: pd.DataFrame, method: str = 'pearson'):
    corr = correlation_matrix(data, method=method)
    if corr.empty:
        return None
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                    title=f"Matriz de Correlación ({method.capitalize()})")
    return fig