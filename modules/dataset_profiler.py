import pandas as pd
import numpy as np


def analizar_dataframe(df):

    info = {}

    for col in df.columns:
        n_unicos = df[col].nunique(dropna=True)
        ratio_unicos = n_unicos / len(df)

        tipo = "desconocido"

        # detectar ID
        if ratio_unicos > 0.95:
            tipo = "id"

        # texto / categórico: soportar object, string (StringDtype) y category
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            if n_unicos <= 20:
                tipo = "categorico_bajo"
            else:
                tipo = "categorico_alto"

        # numerico
        elif pd.api.types.is_numeric_dtype(df[col]):
            valores = df[col].dropna()

            if len(valores) == 0:
                tipo = "numerico_continuo"
            elif np.all(np.mod(valores, 1) == 0) and n_unicos <= 20:
                tipo = "numerico_discreto"
            else:
                tipo = "numerico_continuo"

        # Fallback: si después de todas las comprobaciones sigue 'desconocido'
        # y no es numérica pero tiene pocos únicos, lo tratamos como categórico_bajo.
        if tipo == "desconocido":
            if not pd.api.types.is_numeric_dtype(df[col]) and n_unicos <= 10:
                tipo = "categorico_bajo"
            elif pd.api.types.is_numeric_dtype(df[col]) and n_unicos <= 10:
                # si es numérica pero todos los valores son enteros, la tratamos como discreta
                valores_check = df[col].dropna()
                if not valores_check.empty and np.all(np.mod(valores_check, 1) == 0):
                    tipo = "numerico_discreto"

        # Tipos de Python presentes en la columna (muestra) y primeros valores (para debugging)
        tipos_muestra = df[col].dropna().apply(lambda x: type(x).__name__).value_counts().to_dict()
        primeros_valores = df[col].dropna().unique()[:10].tolist()

        # Convertimos a strings para evitar problemas de serialización / pyarrow en Streamlit
        tipos_muestra_str = str(tipos_muestra)
        primeros_valores_str = ', '.join(map(lambda v: str(v), primeros_valores))

        info[col] = {
            "tipo": tipo,
            "n_unicos": n_unicos,
            "ratio_unicos": ratio_unicos,
            "dtype": str(df[col].dtype),
            "sample_types": tipos_muestra_str,
            "sample_values": primeros_valores_str
        }

    return info