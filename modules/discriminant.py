import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score

def ejecutar_analisis_discriminante(df, columna_objetivo):

    #Autores: Yara y Sebastian

    print("--- SISTEMA DE ANÁLISIS DISCRIMINANTE  ---")

    # 1. VALIDACIÓN DE INTEGRIDAD
    # Revisa si hay columnas sin nombre o nombres vacíos
    if df.columns.isnull().any() or any(c.strip() == "" for c in df.columns):
        return {"error": "CRÍTICO: El dataset contiene columnas sin nombre. Proceso abortado."}

    # 2. SELECCIÓN DE ENFOQUE (Interactividad para el usuario)
    # Identificamos variables numéricas candidatas (excluyendo la objetivo)
    posibles_predictores = df.select_dtypes(include=[np.number]).columns.tolist()
    if columna_objetivo in posibles_predictores:
        posibles_predictores.remove(columna_objetivo)
    
    print(f"\nVariables disponibles para el análisis: {posibles_predictores}")
    seleccion = input("Escriba las variables para su enfoque (separadas por coma) o presione Enter para usar todas: ")
    
    if seleccion.strip() == "":
        X_cols = posibles_predictores
    else:
        # Limpiamos espacios en blanco de la entrada del usuario
        X_cols = [c.strip() for c in seleccion.split(",") if c.strip() in posibles_predictores]
        if not X_cols:
            return {"error": "Error: Las variables seleccionadas no son válidas o no existen."}

    try:
        # 3. PREPARACIÓN Y ENTRENAMIENTO
        X = df[X_cols]
        y = df[columna_objetivo]
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        
        # 4. EXTRACCIÓN DE RESULTADOS 
        y_pred = lda.predict(X)
        
        # Función Discriminante Lineal (Coeficientes y Constantes)
        coeficientes = pd.DataFrame(lda.coef_, columns=X_cols, index=lda.classes_)
        constante = pd.DataFrame(lda.intercept_, columns=['Constante'], index=lda.classes_)
        tabla_funciones = pd.concat([constante, coeficientes], axis=1)
        
        # Matriz de Confusión (Summary of Classification)
        matriz_m = confusion_matrix(y, y_pred)
        df_matriz = pd.DataFrame(matriz_m, 
                                index=[f"Real {c}" for c in lda.classes_], 
                                columns=[f"Pred {c}" for c in lda.classes_])
        
        # Proporción de aciertos (Accuracy)
        precision = accuracy_score(y, y_pred)
        
        # 5. DIAGNÓSTICO DE ERRORES (Observaciones con asterisco **)
        # Identifica qué filas fueron mal clasificadas
        analisis_casos = df.copy()
        analisis_casos['PREDICCIÓN'] = y_pred
        mal_clasificados = analisis_casos[analisis_casos[columna_objetivo] != y_pred]

        print("\n¡Análisis completado con éxito!")

        return {
            "Metodología": "Análisis Discriminante Lineal (LDA)",
            "Variables_Enfoque": X_cols,
            "Funcion_Discriminante": tabla_funciones,
            "Matriz_Confusion": df_matriz,
            "Precision_Global": f"{precision:.2%}",
            "Casos_Atipicos_Asterisco": mal_clasificados,
            "Centroides_Grupos": pd.DataFrame(lda.means_, columns=X_cols, index=lda.classes_)
        }

    except Exception as e:
        return {"error": f"Error en el cálculo multivariante: {str(e)}"}
