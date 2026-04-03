import pandas as pd
import numpy as np


def analizar_dataframe(df):

    info = {}

    for col in df.columns:
        n_unicos = df[col].nunique(dropna=True)
        ratio_unicos = n_unicos / len(df)

        dtype = df[col].dtype

        tipo = "desconocido"

        # detectar ID
        if ratio_unicos > 0.95:
            tipo = "id"

        # texto
        elif dtype == "object":
            if n_unicos <= 20:
                tipo = "categorico_bajo"
            else:
                tipo = "categorico_alto"

        # numerico
        elif pd.api.types.is_numeric_dtype(df[col]):
            valores = df[col].dropna()

            if len(valores) == 0:
                tipo = "numerico_continuo"
            elif np.all(np.mod(valores,1)==0) and n_unicos <= 20:
                tipo = "numerico_discreto"
            else:
                tipo = "numerico_continuo"

        info[col] = {
            "tipo": tipo,
            "n_unicos": n_unicos,
            "ratio_unicos": ratio_unicos
        }

    return info