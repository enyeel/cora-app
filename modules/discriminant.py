import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def ejecutar_analisis_discriminante(df_limpio, columna_objetivo):
    """
    Yara y Sebastian: 
    """
    try:
        # 1. Separar variables numéricas (X) y la variable objetivo (y)
        X = df_limpio.select_dtypes(include=[np.number])
        y = df_limpio[columna_objetivo]
        
        # 2. Inicializar el modelo de Scikit-Learn 
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        
        # 3. Extraer resultados clave
        # Calculamos los centroides (medias)
        centroides = pd.DataFrame(lda.means_, columns=X.columns, index=lda.classes_)
        
        # Calculamos la varianza explicada (qué tanto discriminan las funciones)
        varianza = lda.explained_variance_ratio_
        
       
        resultados = {
            "centroides": centroides,
            "varianza_explicada": varianza,
            "columnas_usadas": list(X.columns),
            "clases": list(lda.classes_)
        }
        
        return resultados

    except Exception as e:
        return {"error": str(e)}