import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# 🩻 MÓDULO DE DETECCIÓN (RAYOS X)
# ==========================================

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
def codificar_categoricos(df):
    """Detecta columnas de texto y aplica One-Hot Encoding."""
    df_temp = df.copy()
    columnas_texto = df_temp.select_dtypes(include=['object']).columns.tolist()

    if len(columnas_texto) > 0:
        # PANDAS MAGIC: Le decimos directamente que queremos 1s y 0s con dtype=int
        df_temp = pd.get_dummies(df_temp, columns=columnas_texto, dtype=int)

    return df_temp

# ==========================================
# FUNCION 5: ESTANDARIZACIÓN Z-SCORE
# ==========================================
def estandarizar_zscore(df, columnas_excluir=['ID_Usuario']):
    """Aplica Z-Score a las columnas numéricas que no sean binarias o IDs."""
    df_temp = df.copy()
    scaler = StandardScaler()

    # Solo agarramos numéricas
    cols_numericas = df_temp.select_dtypes(include=[np.number]).columns.tolist()

    # Filtramos: Quitamos el ID y quitamos las columnas que solo tienen 1s y 0s (Dummies)
    cols_a_estandarizar = [col for col in cols_numericas
                           if col not in columnas_excluir and not set(df_temp[col].dropna().unique()).issubset({0, 1})]

    if len(cols_a_estandarizar) > 0:
        df_temp[cols_a_estandarizar] = scaler.fit_transform(df_temp[cols_a_estandarizar])

    return df_temp
