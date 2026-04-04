import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

# ==========================================
# 🩻 MÓDULO DE DETECCIÓN (RAYOS X)
# ==========================================

def detectar_anomalias_estructurales(df):
    """
    Escanea el DataFrame en busca de columnas 100% vacías, columnas zombie (>60% nulos)
    y columnas que deberían ser numéricas pero tienen textos escondidos (coerción).
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

        # Detectar coerción
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
    """Busca outliers numéricos. Devuelve un mapa de celdas atípicas y las columnas afectadas."""
    cols_num = df.select_dtypes(include=[np.number]).columns
    
    # Creamos un mapa vacío (todo en Falso)
    mapa_outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    cols_afectadas = []
    
    for col in cols_num:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        Lim_inf = Q1 - 1.5 * IQR
        Lim_sup = Q3 + 1.5 * IQR
        
        # Marcamos como True las celdas que se salen de los límites
        mascara = (df[col] < Lim_inf) | (df[col] > Lim_sup)
        mapa_outliers[col] = mascara
        
        if mascara.any(): # Si encontró al menos un outlier en esta columna
            cols_afectadas.append(col)
            
    return mapa_outliers, cols_afectadas

def detectar_webones(df, umbral_varianza=0.0):
    """
    Busca webones agrupando inteligentemente las columnas que comparten 
    el mismo Mínimo y Máximo (Ej. escalas del 1 al 5).
    """
    # 1. Filtramos las numéricas y quitamos el ID
    cols_num = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'ID_Usuario']
    
    grupos_likert = {}
    
    # 2. Agrupamos por Min y Max (ignorando los nulos)
    for col in cols_num:
        # Si la columna está totalmente vacía, la ignoramos
        if df[col].dropna().empty:
            continue
            
        # Sacamos min y max ignorando los NaNs, y forzamos a entero para evitar el "1.0"
        minimo = int(df[col].min(skipna=True))
        maximo = int(df[col].max(skipna=True))
        
        etiqueta_rango = f"Rango_{minimo}_a_{maximo}"
        
        if etiqueta_rango not in grupos_likert:
            grupos_likert[etiqueta_rango] = []
        grupos_likert[etiqueta_rango].append(col)

    # 3. Filtramos los grupos que tengan 3 o más preguntas
    grupos_encuesta = {rango: columnas for rango, columnas in grupos_likert.items() if len(columnas) >= 3}
    
    # Preparamos una lista de falsos (nadie es webón hasta que se demuestre lo contrario)
    filas_webones = pd.Series(False, index=df.index)
    
    # 4. Calculamos la varianza SOLO en los grupos detectados (Ej. Pregunta 1 a 5)
    for rango, columnas in grupos_encuesta.items():
        # Varianza horizontal (axis=1)
        varianza = df[columnas].var(axis=1)
        
        # Marcamos como True a los que tengan varianza nula
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
