import streamlit as st
import pandas as pd

# Aquí vas a importar las funciones de tus compas cuando las tengan listas.
# Por ahora las dejo comentadas para que no te marque error.
# from modules.descriptiva import calcular_descriptiva
# from modules.factorial import hacer_factorial
# from modules.discriminante import hacer_discriminante  # Quitado: no se usa y está mal escrito
# from modules.conglomerados import hacer_conglomerados

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Software de Estadística", page_icon="📊", layout="wide")

# --- FUNCIÓN DE LIMPIEZA (Tu otra chamba) ---
def limpiar_datos(df):
    # Aquí meterás tu lógica: quitar nulos, cambiar tipos de datos, etc.
    # Por ahora, solo regresamos el mismo dataframe.
    df_limpio = df.dropna() 
    return df_limpio

# --- EL MENÚSITO OBVIS (Barra Lateral) ---
st.sidebar.title("⚙️ Menú de Opciones")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Selecciona un módulo:",
    ("1. Carga y Limpieza de BD", 
     "2. Estadística Descriptiva", 
     "3. Análisis Factorial", 
     "4. Análisis Discriminante", 
     "5. Análisis de Conglomerados")
)

st.sidebar.markdown("---")
st.sidebar.info("Proyecto Final - Equipo de 8 🚀")

# --- PANTALLA PRINCIPAL ---
st.title("📊 Software de Análisis Estadístico")

# Inicializar la variable del dataframe en la memoria (session_state)
if 'df_limpio' not in st.session_state:
    st.session_state.df_limpio = None

# --- LÓGICA DE LAS PANTALLAS ---

if menu == "1. Carga y Limpieza de BD":
    st.header("📂 1. Preparación del Archivo")
    st.write("Sube tu base de datos en formato CSV o Excel para comenzar.")
    
    # El uploader de archivos
    archivo_subido = st.file_uploader("Sube tu archivo aquí", type=["csv", "xlsx"])
    
    if archivo_subido is not None:
        try:
            # Leer el archivo dependiendo de la extensión
            if archivo_subido.name.endswith('.csv'):
                df = pd.read_csv(archivo_subido)
            else:
                df = pd.read_excel(archivo_subido)
                
            st.success("¡Archivo cargado con éxito!")
            st.write("Vista previa de los datos originales:")
            st.dataframe(df.head())
            
            # Botón para limpiar
            if st.button("🧹 Limpiar Base de Datos"):
                df_limpio = limpiar_datos(df)
                st.session_state.df_limpio = df_limpio # Lo guardamos en memoria
                st.success("¡Base de datos limpia y lista para usar en los demás módulos!")
                st.dataframe(st.session_state.df_limpio.head())
                
        except Exception as e:
            st.error(f"Hubo un error al leer el archivo: {e}")

# --- VALIDACIÓN: NO DEJAR AVANZAR SI NO HAY DATOS ---
elif st.session_state.df_limpio is None:
    st.warning("⚠️ Primero debes ir a 'Carga y Limpieza de BD' y subir un archivo.")

# --- MÓDULOS DE TUS COMPAS ---
else:
    df_actual = st.session_state.df_limpio # Sacamos el df limpio de la memoria

    if menu == "2. Estadística Descriptiva":
        st.header("📈 2. Estadística Descriptiva")
        st.write("Aquí van los resultados de Carlos y Fani.")
        
        # Así se verá cuando conectes sus funciones:
        # resultados = calcular_descriptiva(df_actual)
        # st.write(resultados)
        
        st.dataframe(df_actual.describe()) # Un placeholder temporal

    elif menu == "3. Análisis Factorial":
        st.header("🧩 3. Análisis Factorial")
        st.write("Aquí van los resultados de Karina y Emiliano.")
        # matriz, varianza = hacer_factorial(df_actual)
        # st.plotly_chart(matriz)

    elif menu == "4. Análisis Discriminante":
        st.header("🎯 4. Análisis Discriminante")
        st.write("El análisis discriminante está disponible en la página '3. Disriminant' del menú lateral.")

    elif menu == "5. Análisis de Conglomerados":
        st.header("🌌 5. Análisis de Conglomerados")
        st.write("Aquí van los resultados de Fernando y Emilio.")
        # dendrograma = hacer_conglomerados(df_actual)
        # st.pyplot(dendrograma)