# ============================================================
# Predicción de PM2.5 a 24 horas con redes neuronales MLP
# ============================================================

import hashlib
import io
import warnings

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ------------------------------------------------------------
# 1. PALETA Y ESTILO
# ------------------------------------------------------------
# Una sola paleta para la interfaz y las gráficas

FONDO = "#F4F6F7"
TARJETA = "#FFFFFF"
TINTA = "#1B2A33"
TEXTO_SUAVE = "#52626C"
BORDE = "#D5DCE0"
ACENTO = "#0F5C7A"
ACENTO_OSCURO = "#0B455C"
CONTRASTE = "#C2691F"

st.set_page_config(
    page_title="Predicción de PM2.5",
    layout="wide"
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600&family=Source+Serif+4:wght@500;600&display=swap');

    .stApp {{
        background: {FONDO};
        color: {TINTA};
        font-family: 'Source Sans 3', 'Segoe UI', sans-serif;
    }}

    .block-container {{
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: {TINTA} !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }}

    .stApp p, .stApp li, .stApp label, .stApp span {{
        color: {TINTA};
    }}

    [data-testid="stCaptionContainer"] * {{
        color: {TEXTO_SUAVE} !important;
    }}

    /* Tarjetas de métricas */
    [data-testid="stMetric"] {{
        background: {TARJETA};
        border: 1px solid {BORDE};
        border-radius: 6px;
        padding: 16px 18px;
    }}
    [data-testid="stMetricLabel"] * {{
        color: {TEXTO_SUAVE} !important;
    }}
    [data-testid="stMetricValue"] * {{
        color: {TINTA} !important;
    }}

    /* Botones principales: el texto va dentro de un <p>, por eso
       se fuerza el color en todos los elementos hijos. */
    .stButton > button,
    .stDownloadButton > button {{
        background: {ACENTO} !important;
        border: 1px solid {ACENTO} !important;
        border-radius: 6px;
        min-height: 44px;
        font-weight: 600;
    }}
    .stButton > button *,
    .stDownloadButton > button * {{
        color: #FFFFFF !important;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        background: {ACENTO_OSCURO} !important;
        border-color: {ACENTO_OSCURO} !important;
    }}
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible {{
        outline: 3px solid {CONTRASTE};
        outline-offset: 2px;
    }}

    /* Zona de carga de archivos */
    [data-testid="stFileUploaderDropzone"] {{
        background: {TARJETA};
        border: 1px dashed {ACENTO};
        border-radius: 6px;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: {TINTA} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background: {TARJETA} !important;
        border: 1px solid {ACENTO} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button * {{
        color: {ACENTO} !important;
    }}

    /* Selector de modelo */
    div[data-baseweb="select"] > div {{
        background: {TARJETA};
        border-color: {BORDE};
    }}
    div[data-baseweb="select"] * {{
        color: {TINTA} !important;
    }}

    hr {{
        border-color: {BORDE};
    }}
</style>
""", unsafe_allow_html=True)


# Estilo de matplotlib. Se aplica una vez y vale para todas las gráficas.
plt.rcParams.update({
    "figure.facecolor": FONDO,
    "axes.facecolor": TARJETA,
    "axes.edgecolor": BORDE,
    "axes.labelcolor": TEXTO_SUAVE,
    "axes.titlecolor": TINTA,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6EAEC",
    "grid.linewidth": 0.8,
    "xtick.color": TEXTO_SUAVE,
    "ytick.color": TEXTO_SUAVE,
    "font.size": 10,
    "legend.frameon": False,
    "lines.antialiased": True,
})

# Resolución de las imágenes. Con 300 dpi las gráficas se ven
# nítidas aunque Streamlit las estire al ancho de la página.
DPI_GRAFICAS = 300


def mostrar_figura(figura):
    """Dibuja la figura en alta resolución y libera la memoria."""
    figura.tight_layout()
    buffer = io.BytesIO()
    figura.savefig(
        buffer,
        format="png",
        dpi=DPI_GRAFICAS,
        bbox_inches="tight",
        facecolor=figura.get_facecolor()
    )
    plt.close(figura)
    buffer.seek(0)
    st.image(buffer, use_container_width=True)


# ------------------------------------------------------------
# 2. DEFINICIÓN DE ENTRADAS Y OBJETIVO
# ------------------------------------------------------------

INPUT_COLUMNS = [
    "TEMP", "PRES", "DEWP", "Iws",
    "hour", "day",
    "pm25_t", "pm25_t-1", "pm25_t-2", "pm25_t-3"
]

TARGET = "pm25_t+24"

# Las arquitecturas tienen diez entradas y una salida.
# La diferencia está en el número de neuronas ocultas.
CONFIGURACIONES = {
    "Modelo propio (10:16:8:1)": (16, 8),
    "Modelo de referencia (10:20:10:1)": (20, 10)
}


# ------------------------------------------------------------
# 3. ENCABEZADO
# ------------------------------------------------------------

st.title("Predicción de PM2.5 a 24 horas")
st.write(
    "Comparación de dos redes neuronales multicapa que estiman la "
    "concentración de partículas finas PM2.5 con un día de anticipación "
    "a partir de variables meteorológicas y del historial reciente."
)

st.divider()


# ------------------------------------------------------------
# 4. CARGA DEL CONJUNTO DE DATOS
# ------------------------------------------------------------

izquierda, derecha = st.columns([1.4, 1])

with izquierda:
    st.subheader("Datos de entrada")
    archivo = st.file_uploader(
        "Selecciona el CSV del proyecto",
        type=["csv"]
    )

with derecha:
    st.subheader("Qué se analiza")
    st.write(
        "Variables meteorológicas, hora, día e historial de PM2.5. "
        "El objetivo es estimar el valor de la columna `pm25_t+24`."
    )

if archivo is None:
    st.info("Carga el archivo CSV para preparar el análisis.")
    st.stop()

# Identificamos el archivo por su contenido para no mezclar
# los resultados de dos archivos distintos.
contenido = archivo.getvalue()
id_archivo = hashlib.sha256(contenido).hexdigest()

try:
    df = pd.read_csv(io.BytesIO(contenido))
    df.columns = df.columns.str.strip()
except Exception as error:
    st.error(f"No se pudo leer el CSV: {error}")
    st.stop()

columnas_requeridas = INPUT_COLUMNS + [TARGET]
faltantes = [c for c in columnas_requeridas if c not in df.columns]

if faltantes:
    st.error("Faltan columnas necesarias: " + ", ".join(faltantes))
    st.stop()

# Seleccionamos solo las columnas definidas para el experimento.
datos = df[columnas_requeridas].copy()

# Validamos los datos sin eliminar filas silenciosamente.
for columna in columnas_requeridas:
    datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

if datos.isna().any().any():
    st.error(
        "Hay valores vacíos o no numéricos en las variables seleccionadas. "
        "Revisa el archivo antes de entrenar."
    )
    st.stop()

if not np.isfinite(datos.to_numpy()).all():
    st.error("Se detectaron valores infinitos. Revisa el CSV.")
    st.stop()

if len(datos) < 20:
    st.error("El archivo contiene muy pocas observaciones para este experimento.")
    st.stop()

# Si cambia el archivo, descartamos los resultados anteriores.
if st.session_state.get("id_archivo") != id_archivo:
    st.session_state["id_archivo"] = id_archivo
    st.session_state["resultados"] = None


# ------------------------------------------------------------
# 5. RESUMEN DEL CONJUNTO
# ------------------------------------------------------------

st.divider()
st.subheader("Resumen del conjunto de datos")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Observaciones", f"{len(datos):,}")
k2.metric("Variables de entrada", len(INPUT_COLUMNS))
k3.metric("Objetivo", "PM2.5 a 24 h")
k4.metric("Valores faltantes", f"{int(datos.isna().sum().sum())}")

with st.expander("Ver una muestra de los datos"):
    st.dataframe(datos.head(8), use_container_width=True)


# ------------------------------------------------------------
# 6. ENTRENAMIENTO DE LOS DOS MODELOS
# ------------------------------------------------------------

st.divider()
st.subheader("Entrenamiento")

st.write(
    "Ambas redes se entrenan con las mismas observaciones y el mismo "
    "preprocesamiento para que la comparación sea justa."
)

if st.button("Entrenar modelos", use_container_width=True):

    # Dividimos en orden cronológico (80 % entrenamiento, 20 % prueba)
    # sin mezclar las observaciones, porque son una secuencia temporal.
    corte = int(len(datos) * 0.80)

    X = datos[INPUT_COLUMNS]
    y = datos[TARGET]

    X_train = X.iloc[:corte]
    X_test = X.iloc[corte:]
    y_train = y.iloc[:corte]
    y_test = y.iloc[corte:]

    # El escalador aprende mínimos y máximos únicamente del
    # entrenamiento. Después aplica esa transformación a la prueba.
    escalador_X = MinMaxScaler()
    X_train_norm = escalador_X.fit_transform(X_train)
    X_test_norm = escalador_X.transform(X_test)

    # Normalizamos también el objetivo para facilitar el aprendizaje.
    escalador_y = MinMaxScaler()
    y_train_norm = escalador_y.fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    ).ravel()

    resultados = {}

    barra = st.progress(0, text="Preparando el entrenamiento...")

    for indice, (nombre, capas) in enumerate(CONFIGURACIONES.items()):

        modelo = MLPRegressor(
            hidden_layer_sizes=capas,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=300,
            random_state=42,
            early_stopping=False,
            tol=1e-5,
            n_iter_no_change=20
        )

        # Registramos advertencias para poder detectar posibles
        # problemas de convergencia sin ocultar los resultados.
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            modelo.fit(X_train_norm, y_train_norm)

        prediccion_norm = modelo.predict(X_test_norm)

        # Recuperamos las unidades originales de PM2.5.
        prediccion = escalador_y.inverse_transform(
            prediccion_norm.reshape(-1, 1)
        ).ravel()

        # Las métricas se calculan en la escala original.
        mse = mean_squared_error(y_test, prediccion)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, prediccion)

        resultados[nombre] = {
            "modelo": modelo,
            "prediccion": prediccion,
            "real": y_test.to_numpy(),
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "iteraciones": modelo.n_iter_,
            "perdida": modelo.loss_curve_,
            "convergencia": [
                str(a.message) for a in avisos
                if "converg" in str(a.message).lower()
            ]
        }

        barra.progress(
            (indice + 1) / len(CONFIGURACIONES),
            text=f"Entrenando {nombre}..."
        )

    barra.empty()

    # Guardamos todo para que el selector de modelo no obligue
    # a entrenar de nuevo cada vez que cambia su selección.
    st.session_state["resultados"] = resultados
    st.session_state["corte"] = corte
    st.session_state["datos_evaluacion"] = datos[TARGET].iloc[corte:].to_numpy()

    st.success("Entrenamiento terminado. Ya puedes explorar los resultados.")


# ------------------------------------------------------------
# 7. SELECTOR Y COMPARACIÓN
# ------------------------------------------------------------

resultados = st.session_state.get("resultados")

if resultados is None:
    st.info(
        "Cuando quieras, pulsa «Entrenar modelos». "
        "La comparación y las gráficas aparecerán aquí."
    )
    st.stop()

st.divider()
st.subheader("Resultados")

opciones = list(CONFIGURACIONES.keys()) + ["Comparar ambos modelos"]

seleccion = st.selectbox(
    "Modelo que deseas analizar",
    options=opciones,
    index=0
)

if seleccion == "Comparar ambos modelos":
    # En la vista comparativa mostramos las métricas de ambas redes.
    tabla = pd.DataFrame({
        nombre: {
            "MSE": r["MSE"],
            "RMSE": r["RMSE"],
            "MAE": r["MAE"],
            "Iteraciones": r["iteraciones"]
        }
        for nombre, r in resultados.items()
    }).T

    st.markdown("#### Comparación de desempeño")
    st.dataframe(
        tabla.style.format({
            "MSE": "{:.3f}",
            "RMSE": "{:.3f}",
            "MAE": "{:.3f}",
            "Iteraciones": "{:.0f}"
        }),
        use_container_width=True
    )

    # La red con menor RMSE se utiliza para las gráficas individuales.
    nombre_activo = min(
        resultados,
        key=lambda nombre: resultados[nombre]["RMSE"]
    )
    st.caption(
        f"Las gráficas de abajo muestran el {nombre_activo}, que tiene el "
        "menor RMSE. Selecciona cada arquitectura para inspeccionarla."
    )

else:
    nombre_activo = seleccion

activo = resultados[nombre_activo]
y_real = activo["real"]
y_pred = activo["prediccion"]

# Métricas destacadas, con una explicación breve de cada indicador.
st.markdown(f"#### {nombre_activo}")

m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{activo['RMSE']:.2f}")
m2.metric("MAE", f"{activo['MAE']:.2f}")
m3.metric("MSE", f"{activo['MSE']:.2f}")

with st.expander("¿Cómo interpretar estas métricas?"):
    st.markdown("""
    - **RMSE:** error que penaliza especialmente los errores grandes.
    - **MAE:** magnitud promedio del error absoluto.
    - **MSE:** promedio del error cuadrático.

    Para las tres métricas, un valor menor indica menor error en este
    conjunto de prueba. RMSE y MAE se expresan en las unidades originales
    de la variable PM2.5; MSE, en unidades al cuadrado.
    """)


# ------------------------------------------------------------
# 8. GRÁFICAS DEL MODELO SELECCIONADO
# ------------------------------------------------------------

st.divider()
st.subheader("Comportamiento de las predicciones")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

# Gráfica izquierda: valores observados frente a predicciones.
ax[0].scatter(
    y_real, y_pred,
    s=10, alpha=0.30,
    color=ACENTO,
    edgecolors="none"
)

limite_min = min(np.min(y_real), np.min(y_pred))
limite_max = max(np.max(y_real), np.max(y_pred))

ax[0].plot(
    [limite_min, limite_max],
    [limite_min, limite_max],
    linestyle="--",
    color=CONTRASTE,
    linewidth=1.6,
    label="Predicción perfecta"
)

ax[0].set_title("Real frente a predicho")
ax[0].set_xlabel("PM2.5 real")
ax[0].set_ylabel("PM2.5 predicho")
ax[0].legend(loc="upper left")

# Gráfica derecha: evolución de la pérdida de entrenamiento.
ax[1].plot(
    activo["perdida"],
    color=ACENTO,
    linewidth=2
)

ax[1].set_title("Pérdida durante el entrenamiento")
ax[1].set_xlabel("Iteración")
ax[1].set_ylabel("Pérdida")

mostrar_figura(fig)

st.caption(
    "En el gráfico de dispersión, los puntos más cercanos a la línea "
    "diagonal representan predicciones más próximas a los valores reales. "
    "La pérdida corresponde al entrenamiento y no sustituye las métricas "
    "calculadas sobre los datos de prueba."
)



# ------------------------------------------------------------
# 10. SERIE TEMPORAL REAL Y PREDICHA
# ------------------------------------------------------------

st.divider()
st.subheader("Seguimiento de la predicción")

cantidad = min(300, len(y_real))

fig_s, ax_s = plt.subplots(figsize=(13, 4))

ax_s.plot(
    y_real[:cantidad],
    color=TINTA,
    linewidth=1.5,
    label="Valor real"
)

ax_s.plot(
    y_pred[:cantidad],
    color=CONTRASTE,
    linewidth=1.5,
    label="Predicción"
)

ax_s.set_title(f"Primeras {cantidad} observaciones de prueba")
ax_s.set_xlabel("Orden de observación")
ax_s.set_ylabel("PM2.5")
ax_s.legend(loc="upper right")

mostrar_figura(fig_s)

# Advertimos si un modelo llegó al máximo de iteraciones.
if activo["iteraciones"] >= 300 or activo["convergencia"]:
    st.warning(
        "El modelo seleccionado podría no haber convergido completamente. "
        "Revisa el número de iteraciones y las advertencias antes de "
        "presentar conclusiones."
    )

# Exportamos la comparación entre valores reales y predichos.
exportacion = pd.DataFrame({
    "PM25_real": y_real,
    "PM25_predicho": y_pred,
    "Error": y_real - y_pred
})

st.download_button(
    "Descargar resultados del modelo seleccionado",
    data=exportacion.to_csv(index=False).encode("utf-8"),
    file_name="resultados_pm25.csv",
    mime="text/csv",
    use_container_width=True
)

st.divider()
st.caption("Proyecto académico de redes neuronales multicapa.")
