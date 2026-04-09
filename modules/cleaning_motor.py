import pandas as pd
import numpy as np

def aplicar_estructural(df, config):
    df_est = df.copy()
    if config.get("estructural", {}).get("enabled", True):
        cols_to_drop = config["estructural"].get("drop_cols", [])
        cols_to_coerce = config["estructural"].get("coerce_cols", [])

        # Drop unnecessary columns
        cols_to_drop_valid = [c for c in cols_to_drop if c in df_est.columns]
        if cols_to_drop_valid:
            df_est = df_est.drop(columns=cols_to_drop_valid)

        # Convert non-numeric values to NaN using forced type coercion
        for col in cols_to_coerce:
            if col in df_est.columns:
                df_est[col] = pd.to_numeric(df_est[col], errors='coerce')

    return df_est

def aplicar_outliers(df, mapa_outliers, config):
    df_out = df.copy()
    if config.get("outliers", {}).get("enabled", False):
        acciones_out = config["outliers"].get("acciones_por_fila", {})
        for id_fila, accion in acciones_out.items():
            if accion == "Eliminar fila completa" and id_fila in df_out.index:
                df_out = df_out.drop(index=id_fila)
            elif accion == "Neutralizar valor (NaN)":
                for col in df_out.columns:
                    if pd.api.types.is_numeric_dtype(df_out[col]):
                        if id_fila in mapa_outliers.index and mapa_outliers.at[id_fila, col]:
                            df_out.at[id_fila, col] = np.nan
    return df_out

def aplicar_webones(df, config):
    df_out = df.copy()
    if config.get("webones", {}).get("enabled", False):
        acciones_web = config["webones"].get("acciones_por_fila", {})
        for id_fila, accion in acciones_web.items():
            if accion == "Eliminar fila completa" and id_fila in df_out.index:
                df_out = df_out.drop(index=id_fila)
    return df_out

def aplicar_nulos(df, config):
    df_out = df.copy()
    if config.get("imputacion", {}).get("enabled", False):
        estrategia_global = config["imputacion"].get("estrategia_global", "Media")
        acciones_col = config["imputacion"].get("acciones_por_columna", {})
        cols_actuales_con_nulos = df_out.columns[df_out.isnull().any()].tolist()
        
        for col in cols_actuales_con_nulos:
            accion_elegida = acciones_col.get(col, "Usar Estrategia Global")
            accion_real = f"Imputar por {estrategia_global}" if accion_elegida == "Usar Estrategia Global" else accion_elegida
                
            if accion_real == "Eliminar filas con nulos":
                df_out = df_out.dropna(subset=[col])
                continue
            elif accion_real == "Ignorar (Dejar nulo)":
                continue
            
            es_numerica = pd.api.types.is_numeric_dtype(df_out[col])
            
            if not es_numerica:
                valor_imputacion = df_out[col].mode()[0]
                df_out[col] = df_out[col].fillna(valor_imputacion)
            else:
                valor_imputacion = df_out[col].mean() if accion_real == "Imputar por Media" else df_out[col].median()
                valores_originales_limpios = df_out[col].dropna()
                es_discreta = np.all(np.mod(valores_originales_limpios, 1) == 0)
                
                df_out[col] = df_out[col].fillna(valor_imputacion)
                if es_discreta:
                    df_out[col] = np.round(df_out[col]).astype(int)
    return df_out