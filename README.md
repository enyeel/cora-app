# 📊 Proyecto de Estadística - Guía para el Equipo

¡Qué onda banda! Aquí están las reglas del juego para armar este software en 4 semanas sin matarnos entre nosotros. Somos 8, así que si nos organizamos, sale al puro centavo.

La arquitectura es sencilla: **Yo (Alex) armo la interfaz web y les paso los datos limpios. Ustedes hacen la magia matemática en sus módulos y me regresan los resultados para que yo los muestre.**

---

## 🏗️ 1. Arquitectura General y Carpetas
Vamos a usar **Streamlit** para que se vea como página web sin tener que pelearnos con HTML/CSS. 

Nadie toca mi archivo principal (`app.py`). Cada equipo va a trabajar en su propio archivo de Python dentro de la carpeta `modules/`. La estructura del proyecto es esta:

```code
proyecto_estadistica/
│
├── app.py                 # El front con Streamlit (Chamba de Alex)
├── requirements.txt       # Las librerías que vamos a usar
│
├── data/                  # Aquí metan sus CSVs para hacer pruebas
│
└── modules/               # LA CHAMBA DE USTEDES:
    ├── __init__.py        # NO BORRAR. Archivo vacío para que funcionen los imports
    ├── descriptiva.py     # Carlos y Fani
    ├── factorial.py       # Karina y Emiliano
    ├── discriminante.py   # Lora y Sebastian
    └── conglomerados.py   # Fernando y Emilio
```

## 🤝 2. ¿Cómo nos conectamos? (Input y Output)
Yo me encargo de cargar la Base de Datos (.csv o .xlsx), quitar los nulos y dejarla lista.
- **Lo que yo les doy (Input):** Sus funciones principales deben recibir un DataFrame de Pandas ya limpio.
- **Lo que ustedes me regresan (Output):** Su función debe devolver un diccionario, un nuevo DataFrame, o un objeto de gráfica (figura) que yo pueda agarrar y mostrar en la web.


**🚨 LA REGLA DE ORO: ¡PROHIBIDO** usar ```print()``` o ```plt.show()```! Si usan eso, se imprime en la consola, no en la página web. Todo debe tener un return.


Ejemplo de cómo debe verse su código en su archivo de módulo:

```Python
import pandas as pd
# import plotly.express as px ... etc

def hacer_calculos(df_limpio):
    # Aquí hacen todo su desmadre
    resultados_tabla = df_limpio.describe()
    
    # Regresan la tabla o gráfica para que yo la muestre
    return resultados_tabla
```

## 🛠️ 3. Tareas y Librerías por Equipo

### 📈 Carlos y Fani: Estadística Descriptiva (```modules/descriptive.py```)
- **Misión:** Sacar medidas de tendencia central, dispersión, forma y tablas de frecuencias. Gráficos básicos.
- **Librerías recomendadas:** ```pandas```, ```scipy.stats```.
- **Gráficas:** Usen ```plotly.express``` o ```seaborn```. Recuerden devolver la figura (el objeto fig).

### 🧩 Karina y Emiliano: Análisis Factorial (```modules/factorial.py```)
- **Misión:** Reducción de dimensiones. Matriz de correlaciones, prueba KMO, Bartlett, matriz de cargas factoriales y varianza explicada.
- **Librerías recomendadas:** ```factor_analyzer``` y ```pandas```.

### 🎯 Yara y Sebastian: Análisis Discriminante (```modules/discriminant.py```)
- **Misión:** Clasificar o ver qué variables discriminan mejor entre grupos.
- **Librerías recomendadas:** ```scikit-learn``` (```LinearDiscriminantAnalysis```).
- **Ojo:** Su función va a necesitar recibir el DataFrame limpio y la variable objetivo.

### 🌌 Fernando y Emilio: Conglomerados (```modules/clustering.py```)
- **Misión:** Agrupar los datos. Modelo No Jerárquico (K-Means) y Jerárquico con Dendrograma.
- **Librerías recomendadas:** ```scikit-learn``` (```KMeans```, ```AgglomerativeClustering```) y ```scipy.cluster.hierarchy```.

## 🚨 4. Flujo de Trabajo (Git)
- Descarguen el repo (git clone).
- Hagan sus commits y push avisando por WhatsApp qué subieron.
