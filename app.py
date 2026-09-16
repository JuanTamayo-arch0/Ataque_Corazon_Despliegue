"""
Despliegue - Predicción de Ataque de Corazón (stroke)
Ejecutar con: streamlit run app.py

- Carga el modelo Knn entrenado en 4_Validacion_Cruzada_Class
- Captura los datos por interfaz (formulario o perfiles de ejemplo)
- Prepara los datos: dummies (2 categorías y multicategoría) + normalización
- Predice, muestra probabilidad (gauge) y recomendaciones personalizadas
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Cargamos el modelo, el encoder de la variable objetivo, las variables (columnas) del entrenamiento
#y el escalador ya ajustado (fit) en el entrenamiento
#IMPORTANTE: el orden del pickle.dump fue [modelo_final, labelencoder, variables, min_max_scaler]

import pickle
filename = 'modelo-cla.pkl'
modelo, labelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

#Cargamos los datos futuros (opcional, si no se usa la interfaz)
#data = pd.read_excel("ataque_corazon-datosFuturos.xlsx")
#data.head()

import streamlit as st

st.set_page_config(page_title="Riesgo Cardiovascular", page_icon="🫀", layout="wide")

#Estilos personalizados: degradado, tarjetas de color, tipografía
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #6a5acd 0%, #ff5c7a 100%);
    padding: 1.8rem 2rem;
    border-radius: 18px;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}
.hero h1 {color: white; margin: 0; font-size: 1.6rem;}
.hero p {color: #f0e9ff; margin: 0.2rem 0 0 0;}
.card {
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-top: 0.8rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
.card-riesgo {background-color: #ffe3e8; border-left: 6px solid #ff4d6d;}
.card-normal {background-color: #e3f8ec; border-left: 6px solid #2ecc71;}
</style>
""", unsafe_allow_html=True)

#Imagen del encabezado: ícono SVG de corazón + línea de pulso (dibujado en código, no depende de internet)
heart_svg = """
<svg width="90" height="70" viewBox="-10 -5 110 100" xmlns="http://www.w3.org/2000/svg">
  <path d="M40,20 C25,0 -5,15 -5,40 C-5,60 20,80 40,95 C60,80 85,60 85,40 C85,15 55,0 40,20 Z"
        fill="#ffffff" opacity="0.92"/>
  <polyline points="0,55 20,55 28,35 36,75 44,45 52,55 90,55"
            fill="none" stroke="#6a5acd" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

#NOTA: cada línea del bloque va sin indentación. Si se indentan 4+ espacios,
#Markdown las interpreta como un bloque de código y se ven como texto plano.
st.markdown(f"""
<div class="hero">
{heart_svg}
<div>
<h1>Predicción de Riesgo de Ataque de Corazón</h1>
<p>Modelo Knn &middot; validación cruzada estratificada &middot; f1-macro &asymp; 0.816</p>
</div>
</div>
""", unsafe_allow_html=True)

#Barra lateral: perfiles de ejemplo para probar rápido la app (personalización)
st.sidebar.header("⚙️ Opciones")
st.sidebar.markdown("Puedes cargar un perfil de ejemplo o llenar el formulario manualmente.")

perfil = st.sidebar.radio(
    "Perfil de ejemplo",
    ["Ninguno (llenar manualmente)", "Bajo riesgo", "Alto riesgo"]
)

valores_por_defecto = {
    "Ninguno (llenar manualmente)": dict(age=45, avg_glucose_level=100.0, hypertension="No",
                                          heart_disease="No", ever_married="Yes", smoking_status="Unknown"),
    "Bajo riesgo": dict(age=28, avg_glucose_level=85.0, hypertension="No",
                         heart_disease="No", ever_married="No", smoking_status="'never smoked'"),
    "Alto riesgo": dict(age=74, avg_glucose_level=210.0, hypertension="Yes",
                         heart_disease="Yes", ever_married="Yes", smoking_status="smokes"),
}
defaults = valores_por_defecto[perfil]

#Formulario con íconos por campo, organizado en dos columnas
opciones_binarias = ["No", "Yes"]
opciones_fumador = ["Unknown", "'formerly smoked'", "'never smoked'", "smokes"]

with st.form("formulario_paciente"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧍 Datos personales")
        age = st.slider("🎂 Edad (age)", min_value=0, max_value=100,
                         value=defaults["age"], step=1)
        ever_married = st.selectbox("💍 ¿Alguna vez se ha casado? (ever_married)",
                                     opciones_binarias, index=opciones_binarias.index(defaults["ever_married"]))
        smoking_status = st.selectbox("🚬 Estado de fumador (smoking_status)",
                                       opciones_fumador, index=opciones_fumador.index(defaults["smoking_status"]))

    with col2:
        st.markdown("#### 🩺 Datos clínicos")
        avg_glucose_level = st.number_input("🩸 Nivel promedio de glucosa (avg_glucose_level)",
                                             min_value=40.0, max_value=300.0,
                                             value=float(defaults["avg_glucose_level"]), step=0.1)
        hypertension = st.selectbox("💢 Hipertensión (hypertension)",
                                     opciones_binarias, index=opciones_binarias.index(defaults["hypertension"]))
        heart_disease = st.selectbox("❤️‍🩹 Enfermedad cardíaca (heart_disease)",
                                      opciones_binarias, index=opciones_binarias.index(defaults["heart_disease"]))

    enviado = st.form_submit_button("🔍 Calcular riesgo")

#Dataframe con los mismos nombres de variables que en el entrenamiento
datos = [[age, avg_glucose_level, hypertension, heart_disease, ever_married, smoking_status]]
data = pd.DataFrame(datos, columns=['age', 'avg_glucose_level', 'hypertension',
                                     'heart_disease', 'ever_married', 'smoking_status'])

#Se realiza la preparación de datos (misma lógica que en 4_Validacion_Cruzada_Class)
data_preparada = data.copy()

predictoras_numericas = ['age', 'avg_glucose_level']
predictoras_categoricas_2cat = ['hypertension', 'heart_disease', 'ever_married']
predictoras_categoricas_multicat = ['smoking_status']

#3 o más categorías -> drop_first=False
data_preparada = pd.get_dummies(data_preparada, columns=predictoras_categoricas_multicat,
                                 drop_first=False, dtype=int)
#2 categorías (drop_first=True en el entrenamiento): con una sola fila, pd.get_dummies
#NO sirve aquí. Al haber una única categoría presente en la fila, get_dummies genera
#0 columnas para esa variable, y el reindex de más abajo la rellenaría siempre con 0,
#sin importar si el usuario marcó "Yes" o "No" (por eso hipertensión/enfermedad
#cardíaca no cambiaban la predicción). Se recrean a mano las columnas que sobrevivieron
#al drop_first en el entrenamiento (hypertension_Yes, heart_disease_Yes, ever_married_Yes).
for col in predictoras_categoricas_2cat:
    data_preparada[f"{col}_Yes"] = (data_preparada[col] == "Yes").astype(int)
    data_preparada = data_preparada.drop(columns=[col])

#Se adicionan las columnas faltantes (dummies de categorías que no salieron en este registro)
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

#Se normalizan las variables numéricas para predecir con Knn (modelo seleccionado)
#En los despliegues no se llama fit, solo transform
data_preparada[predictoras_numericas] = min_max_scaler.transform(data_preparada[predictoras_numericas])

#Hacemos la predicción con el Knn
Y_pred = modelo.predict(data_preparada)
Y_pred_decodificada = labelencoder.inverse_transform(Y_pred)

#Probabilidad de la clase (Knn permite predict_proba) para el gauge y la tarjeta de resultado
proba = modelo.predict_proba(data_preparada)[0]
clases_decodificadas = labelencoder.inverse_transform(modelo.classes_)
if "Yes" in clases_decodificadas:
    prob_riesgo = proba[list(clases_decodificadas).index("Yes")]
else:
    prob_riesgo = proba.max()

data['Prediccion'] = Y_pred_decodificada

#Gauge (velocímetro) de probabilidad, dibujado con matplotlib
def gauge_chart(valor, titulo="Probabilidad de riesgo"):
    fig, ax = plt.subplots(figsize=(4, 2.6))
    valores = [0.33, 0.33, 0.34]
    colores = ['#2ecc71', '#f1c40f', '#ff4d6d']
    ax.pie(valores + [sum(valores)], colors=colores + ['white'],
           startangle=180, counterclock=False,
           wedgeprops=dict(width=0.35, edgecolor='white'))

    angulo = math.radians(180 - valor * 180)
    x, y = 0.65 * math.cos(angulo), 0.65 * math.sin(angulo)
    ax.plot([0, x], [0, y], color='#2c2c2c', linewidth=3, solid_capstyle='round')
    ax.add_patch(plt.Circle((0, 0), 0.06, color='#2c2c2c', zorder=5))

    ax.text(0, -0.15, f"{valor*100:.1f}%", ha='center', va='center', fontsize=20, fontweight='bold')
    ax.set_title(titulo, fontsize=12, pad=10)
    ax.set_ylim(-0.2, 1.1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig


#Tarjeta de resultado (color según el riesgo). Sin indentación en las líneas HTML
#(ver nota más arriba sobre Markdown y bloques de código).
if Y_pred_decodificada[0] == 'Yes':
    st.markdown(f"""
<div class="card card-riesgo">
<h3>⚠️ Riesgo elevado de ataque de corazón</h3>
<p>El modelo estima una probabilidad de <b>{prob_riesgo*100:.1f}%</b> para este perfil.</p>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown(f"""
<div class="card card-normal">
<h3>✅ Riesgo bajo de ataque de corazón</h3>
<p>El modelo estima una probabilidad de <b>{prob_riesgo*100:.1f}%</b> para este perfil.</p>
</div>
""", unsafe_allow_html=True)

#use_container_width=False evita que Streamlit estire la figura al ancho completo
#de la página (eso era lo que hacía ver la aguja "gigante"). La centramos en una
#columna angosta para que quede proporcionada.
_, col_gauge, _ = st.columns([1, 1.2, 1])
with col_gauge:
    st.pyplot(gauge_chart(prob_riesgo), use_container_width=False)

#Recomendaciones personalizadas según los factores de riesgo capturados en el formulario
factores = []
if hypertension == "Yes":
    factores.append("hipertensión")
if heart_disease == "Yes":
    factores.append("antecedente de enfermedad cardíaca")
if smoking_status in ["smokes", "'formerly smoked'"]:
    factores.append("hábito de fumar")
if avg_glucose_level > 125:
    factores.append("nivel de glucosa elevado")
if age > 60:
    factores.append("edad mayor a 60 años")

if factores:
    st.info("🔎 Factores presentes en este perfil: " + ", ".join(factores) +
            ". Esta es una estimación estadística, no un diagnóstico médico.")
else:
    st.info("🔎 No se identificaron factores de riesgo clásicos adicionales en este perfil.")

with st.expander("📊 Ver datos y detalle de la predicción"):
    st.dataframe(data)

#Comparación de modelos evaluados en la validación cruzada (por qué se eligió Knn)
with st.expander("📈 Comparación de modelos (validación cruzada, f1-macro)"):
    comparacion = pd.DataFrame({
        "Modelo": ["Tree", "RF", "Knn (seleccionado)", "NN", "SVM"],
        "f1_macro_test": [0.707, 0.779, 0.816, 0.738, 0.709]
    })
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    colores_barra = ['#b0b0b0', '#b0b0b0', '#6a5acd', '#b0b0b0', '#b0b0b0']
    ax2.bar(comparacion["Modelo"], comparacion["f1_macro_test"], color=colores_barra)
    ax2.set_ylabel("f1-macro (test, CV)")
    ax2.set_ylim(0, 1)
    plt.xticks(rotation=15)
    st.pyplot(fig2)
