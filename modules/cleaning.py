import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

# ====================================================================
# Data Anomaly Detection Module
# ====================================================================

def detectar_anomalias_estructurales(df):
    """
    Scans the DataFrame for completely empty columns, sparse columns (>60% missing),
    and mixed data types in numeric columns.
    """
    reporte = {
        "vacias": [],
        "zombies": [],
        "coercion": {}
    }
    
    for col in df.columns:
        pct_nulos = df[col].isnull().mean()
        
        # Detectar vacías y zombies
        if pct_nulos == 1.0:
            reporte["vacias"].append(col)
        elif pct_nulos >= 0.60:
            reporte["zombies"].append(col)

        # Detect numeric columns with mixed data types
        if not is_numeric_dtype(df[col]):
            s_orig = df[col]
            s_num = pd.to_numeric(s_orig, errors='coerce')
            mask_rebeldes = s_orig.notna() & s_num.isna()
            
            if s_num.notna().any():
                tasa_exito = s_num.notna().sum() / s_orig.notna().sum()
                if tasa_exito > 0.50 and mask_rebeldes.any():
                    reporte["coercion"][col] = df.loc[mask_rebeldes, [col]]
                    
    return reporte

def detectar_outliers(df):
    """Identifies numeric outliers using the IQR method.
    Returns a map of outlier cells and list of affected columns."""
    cols_num = df.select_dtypes(include=[np.number]).columns
    
    # Create an empty map (all False initially)
    mapa_outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    cols_afectadas = []
    
    for col in cols_num:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        Lim_inf = Q1 - 1.5 * IQR
        Lim_sup = Q3 + 1.5 * IQR
        
        # Mark cells outside the outlier boundaries
        mascara = (df[col] < Lim_inf) | (df[col] > Lim_sup)
        mapa_outliers[col] = mascara
        
        if mascara.any():
            cols_afectadas.append(col)
            
    return mapa_outliers, cols_afectadas

def detectar_webones(df, umbral_varianza=0.0):
    """
    Identifies records with zero variance by intelligently grouping columns
    that share the same minimum and maximum values (e.g., Likert scales).
    """
    # Filter numeric columns excluding ID
    cols_num = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'ID_Usuario']
    
    grupos_likert = {}
    
    # Group columns by Min and Max range
    for col in cols_num:
        # Skip if column is completely empty
        if df[col].dropna().empty:
            continue
            
        # Extract min/max ignoring NaN values
        minimo = int(df[col].min(skipna=True))
        maximo = int(df[col].max(skipna=True))
        
        etiqueta_rango = f"Rango_{minimo}_a_{maximo}"
        
        if etiqueta_rango not in grupos_likert:
            grupos_likert[etiqueta_rango] = []
        grupos_likert[etiqueta_rango].append(col)

    # Filter to keep only groups with 3+ items
    grupos_encuesta = {rango: columnas for rango, columnas in grupos_likert.items() if len(columnas) >= 3}
    
    # Initialize series with all False values
    filas_webones = pd.Series(False, index=df.index)
    
    # Calculate variance only for detected grouped items (e.g., Likert scales)
    for rango, columnas in grupos_encuesta.items():
        # Horizontal variance (axis=1)
        varianza = df[columnas].var(axis=1)
        
        # Mark records with zero variance
        filas_webones = filas_webones | (varianza <= umbral_varianza)
        
    return filas_webones

# ==========================================
# FUNCION 1: ONE-HOT ENCODING (Textos a Números)
# ==========================================
def codificar_categoricos_inteligente(df, metadata):

    df_temp = df.copy()

    columnas_onehot = [
        col for col,meta in metadata.items()
        if meta["tipo"] == "categorico_bajo"
    ]

    if len(columnas_onehot) > 0:

        df_temp = pd.get_dummies(
            df_temp,
            columns=columnas_onehot,
            dtype="int8"
        )

    return df_temp

# ==========================================
# FUNCION 5: ESTANDARIZACIÓN Z-SCORE
# ==========================================
def estandarizar_zscore(df, metadata=None, columnas_excluir=None):
    """Aplica Z-Score a las columnas numéricas que no sean binarias o IDs.

    Parámetros:
    - df: DataFrame a transformar.
    - metadata: diccionario devuelto por `analizar_dataframe` (opcional). Si se entrega,
      las columnas cuyo metadata indique `categorico_*` serán excluidas automáticamente.
    - columnas_excluir: lista adicional de columnas a excluir (opcional).

    Retorna un DataFrame con las columnas numéricas seleccionadas estandarizadas.
    """
    df_temp = df.copy()
    scaler = StandardScaler()

    if columnas_excluir is None:
        columnas_excluir = []

    # Si hay metadata, excluimos las columnas categóricas detectadas (por ejemplo dummies)
    cols_excl_from_meta = []
    if metadata is not None and isinstance(metadata, dict):
        for col, meta in metadata.items():
            tipo = meta.get('tipo', '')
            if isinstance(tipo, str) and tipo.startswith('categorico'):
                cols_excl_from_meta.append(col)

    # Solo agarramos numéricas
    cols_numericas = df_temp.select_dtypes(include=[np.number]).columns.tolist()

    # Filtramos: Quitamos IDs, columnas excluidas por metadata y las columnas que son binarias (0/1)
    cols_a_estandarizar = []
    for col in cols_numericas:
        if col in columnas_excluir or col in cols_excl_from_meta:
            continue
        uniques = df_temp[col].dropna().unique()
        # Si la columna es estrictamente 0/1 (dummies), la excluimos
        if set(map(lambda x: int(x) if (isinstance(x, (int, np.integer)) or (isinstance(x, (float, np.floating)) and float(x).is_integer())) else x, uniques)).issubset({0, 1}):
            continue
        cols_a_estandarizar.append(col)

    if len(cols_a_estandarizar) > 0:
        df_temp[cols_a_estandarizar] = scaler.fit_transform(df_temp[cols_a_estandarizar])

    return df_temp
