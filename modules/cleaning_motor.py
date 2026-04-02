import pandas as pd
import numpy as np

def ejecutar_pipeline_maestro(df_original, mapa_outliers, config):
    """
    Recibe el DataFrame intocable, los mapas de detección y el diccionario 
    de decisiones en tiempo real. Ejecuta la limpieza en orden estricto.
    """
    # 1. Hacemos una copia sagrada para trabajar sin romper la original
    df = df_original.copy()
    
    # ==========================================
    # PASO 1: TRATAMIENTO DE OUTLIERS
    # ==========================================
    if config.get("outliers", {}).get("enabled", False):
        acciones = config["outliers"].get("acciones_por_fila", {})
        
        for id_fila, accion in acciones.items():
            if accion == "Eliminar fila completa":
                # Verificamos que la fila siga existiendo antes de borrarla
                if id_fila in df.index:
                    df = df.drop(index=id_fila)
                    
            elif accion == "Neutralizar valor (NaN)":
                # Magia: Buscamos en el mapa_outliers qué columnas específicas 
                # de ESTA fila son las que están marcadas como True (Azules)
                cols_afectadas = mapa_outliers.columns[mapa_outliers.loc[id_fila]]
                
                for col in cols_afectadas:
                    if id_fila in df.index:
                        df.at[id_fila, col] = np.nan
                        
    # ==========================================
    # PASO 2: WEBONES (Próximamente...)
    # ==========================================
    
    # ==========================================
    # PASO 3: IMPUTACIÓN (Próximamente...)
    # ==========================================

    return df