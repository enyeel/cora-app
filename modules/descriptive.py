import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kstest, skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go

def normality_tests(data: pd.DataFrame, column: str) -> pd.DataFrame:
    col_data = data[column].dropna()
    
    # Shapiro-Wilk (subsample si > 5000)
    col_data_shapiro = col_data.sample(5000, random_state=42) if len(col_data) > 5000 else col_data
    shapiro_stat, shapiro_p = shapiro(col_data_shapiro)
    
    # Kolmogorov-Smirnov
    mean, std = np.mean(col_data), np.std(col_data, ddof=1)
    ks_stat, ks_p = kstest(col_data, 'norm', args=(mean, std))
    
    alpha = 0.05
    is_normal = (shapiro_p > alpha) and (ks_p > alpha)
    
    conclusion = "Distribución Normal" if is_normal else "Distribución No Normal"
    test_type = "Paramétrico (Pearson)" if is_normal else "No paramétrico (Spearman)"
    
    return pd.DataFrame({
        "Shapiro-Wilk (p)": [shapiro_p],
        "K-S (p)": [ks_p],
        "Es Normal": [is_normal],
        "Conclusión": [conclusion],
        "Sugerencia": [test_type]
    })

def correlation_matrix(data: pd.DataFrame, method: str = 'pearson', include_categorical: bool = True) -> pd.DataFrame:
    df_temp = data.copy()
    
    if include_categorical:
        # Convertimos objetos/categorías a códigos numéricos para la matriz
        for col in df_temp.select_dtypes(include=['object', 'category']).columns:
            df_temp[col] = df_temp[col].astype('category').cat.codes
            
    numeric_data = df_temp.select_dtypes(include=['number']).dropna(how='all', axis=1)
    if numeric_data.shape[1] < 2:
        return pd.DataFrame()
    
    return numeric_data.corr(method=method)

def correlation_heatmap(data: pd.DataFrame, method: str = 'pearson', include_categorical: bool = True):
    corr = correlation_matrix(data, method=method, include_categorical=include_categorical)
    if corr.empty:
        return None
    
    fig = px.imshow(
        corr, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=f"Matriz de Correlación ({method.capitalize()}) - Incluye Categóricas"
    )
    return fig

# --- Las demás funciones (central_tendency, frequency_table, etc.) se mantienen igual ---
# (Asegúrate de mantener las funciones de gráficas como histogram_from_table que ya tenías)
def frequency_table(data: pd.DataFrame, column: str, bins: int) -> pd.DataFrame:
    col_data = data[column].dropna()
    intervals = pd.cut(col_data, bins=bins)
    freq = intervals.value_counts().sort_index()
    rel_freq = freq / freq.sum()
    class_marks = [((iv.left + iv.right) / 2) for iv in freq.index]
    
    return pd.DataFrame({
        "Interval": freq.index.astype(str),
        "Class Mark": class_marks,
        "Frequency": freq.values,
        "Relative Frequency": rel_freq.values,
        "Cumulative Relative Frequency": rel_freq.cumsum().values
    })

def central_tendency(data, col):
    d = data[col].dropna()
    return pd.DataFrame({"Measure": ["Mean", "Median"], "Value": [np.mean(d), np.median(d)]})

def dispersion_measures(data, col):
    d = data[col].dropna()
    return pd.DataFrame({"Measure": ["Std Dev"], "Value": [np.std(d, ddof=1)]})

def shape_measures(data, col):
    d = data[col].dropna()
    return pd.DataFrame({"Measure": ["Skewness", "Kurtosis"], "Value": [skew(d), kurtosis(d)]})

def position_measures(data, col):
    d = data[col].dropna()
    q = np.percentile(d, [25, 50, 75])
    return pd.DataFrame({"Measure": ["Q1", "Q2", "Q3"], "Value": q})

def interpret_shape(s, k):
    return pd.DataFrame({"Interpretación": ["Analizando asimetría y curtosis..."]})

def histogram_from_table(table, col):
    return px.bar(table, x="Interval", y="Frequency", title=f"Histograma: {col}")

def frequency_polygon(table, col):
    return px.line(table, x="Class Mark", y="Frequency", title=f"Polígono: {col}")

def boxplot(df, col):
    return px.box(df, y=col, title=f"Boxplot: {col}")

def ogive(table, col):
    return px.line(table, x="Class Mark", y="Cumulative Relative Frequency", title=f"Ojiva: {col}")

def scatter_matrix(df):
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] > 1:
        return px.scatter_matrix(nums)
    return None

def categorical_frequency_table(df, col):
    freq = df[col].value_counts().reset_index()
    freq.columns = ['Category', 'Frequency']
    return freq

def plot_categorical_bar(table, col):
    return px.bar(table, x='Category', y='Frequency', title=f"Frecuencia: {col}")
