import pandas as pd
import numpy as np


def analizar_dataframe(df):
    info = {}

    for col in df.columns:
        n_unicos = df[col].nunique(dropna=True)
        total_filas = len(df)
        ratio_unicos = n_unicos / total_filas if total_filas > 0 else 0
        
        tipo = "desconocido"
        valores_check = df[col].dropna()

        # --- 1. DETECCIÓN PARA NUMÉRICOS ---
        if pd.api.types.is_numeric_dtype(df[col]):
            # Si está vacío, por defecto continuo
            if valores_check.empty:
                tipo = "numerico_continuo"
            
            # ¿Tiene decimales? (Si algún valor mod 1 != 0, es flotante real)
            tiene_decimales = not np.all(np.mod(valores_check, 1) == 0)

            if tiene_decimales:
                # Si tiene decimales, es CONTINUO sí o sí (no puede ser ID)
                tipo = "numerico_continuo"
            else:
                # Si son enteros, podrían ser IDs o Discretos
                if ratio_unicos > 0.95 and n_unicos > 20:
                    tipo = "id"
                elif n_unicos <= 20:
                    tipo = "numerico_discreto"
                else:
                    tipo = "numerico_continuo"

        # --- 2. DETECCIÓN PARA CATEGÓRICOS / TEXTO ---
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            if ratio_unicos > 0.95 and n_unicos > 20:
                tipo = "id"
            elif n_unicos <= 20:
                tipo = "categorico_bajo"
            else:
                tipo = "categorico_alto"

        # --- 3. FALLBACK ---
        if tipo == "desconocido":
            if n_unicos <= 10:
                tipo = "categorico_bajo"
            else:
                tipo = "id" if ratio_unicos > 0.80 else "categorico_alto"

        # Metadata extra para debugging
        tipos_muestra = valores_check.apply(lambda x: type(x).__name__).value_counts().to_dict()
        primeros_valores = valores_check.unique()[:10].tolist()

        info[col] = {
            "tipo": tipo,
            "n_unicos": n_unicos,
            "ratio_unicos": ratio_unicos,
            "dtype": str(df[col].dtype),
            "sample_types": str(tipos_muestra),
            "sample_values": ', '.join(map(str, primeros_valores))
        }

    return info