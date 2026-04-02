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
# FUNCION 2: OUTLIERS (Método IQR)
# ==========================================

# ==========================================
# FUNCION 3: FILTRO STRAIGHT-LINING (Webones)
# ==========================================
def eliminar_webones(df, accion='Eliminar', umbral_varianza=0.0):
    """Agrupa preguntas tipo Likert por rangos y elimina/imputa a los que tienen varianza 0."""
    df_temp = df.copy()
    grupos_likert = {}

    # 1. Autodetección de rangos
    for col in df_temp.select_dtypes(include=[np.number]).columns:
        if col != 'ID_Usuario':
            minimo, maximo = df_temp[col].min(), df_temp[col].max()
            etiqueta = f"Rango_{minimo}_{maximo}"
            if etiqueta not in grupos_likert: grupos_likert[etiqueta] = []
            grupos_likert[etiqueta].append(col)

    # Filtramos grupos que sean encuestas (3 o más preguntas)
    grupos_encuesta = {k: v for k, v in grupos_likert.items() if len(v) >= 3}

    # 2. Cazamos webones por cada grupo
    for rango, columnas in grupos_encuesta.items():
        varianza = df_temp[columnas].std(axis=1)

        if accion == 'Eliminar':
            df_temp = df_temp[varianza > umbral_varianza]
        elif accion == 'Imputar':
            # Los convertimos en nulos (Magia nivel Dios)
            df_temp.loc[varianza == 0, columnas] = np.nan

    return df_temp

# ==========================================
# FUNCION 4: IMPUTACIÓN DE NULOS
# ==========================================
def imputar_nulos(df, metodo='Media', config_tipos={}):
    """Rellena NaN con Media/Mediana y redondea variables Discretas de forma segura."""
    df_temp = df.copy()

    # FASE A: Rellenar absolutamente todos los huecos
    for col in df_temp.columns:
        if df_temp[col].isnull().any():
            valor = df_temp[col].mean() if metodo == 'Media' else df_temp[col].median()

            # Salvavidas: Si la columna estaba 100% vacía, valor será NaN. Lo forzamos a 0.
            if pd.isna(valor):
                valor = 0

            df_temp[col] = df_temp[col].fillna(valor)

    # FASE B: Forzar los tipos de datos (Redondear o dejar decimales)
    for col in df_temp.columns:
        tipo = config_tipos.get(col, 'C') # Por defecto Continua 'C'

        if tipo == 'D':
            # Solo forzamos a entero si estamos 100% seguros de que ya no hay nulos
            if not df_temp[col].isnull().any():
                df_temp[col] = np.round(df_temp[col]).astype(int)
        else:
            # Si es continua (o si por alguna extraña razón no la pudo rellenar), se queda flotante
            df_temp[col] = df_temp[col].astype(float)

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