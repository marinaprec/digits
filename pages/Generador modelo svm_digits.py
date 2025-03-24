from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st


# 1. Cargar el dataset de dígitos (8x8 imágenes)
digits = load_digits()
X = digits.data           # Datos: (n_samples, 64)
y = digits.target         # Etiquetas: (n_samples,)

# 2. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Crear y entrenar el modelo SVM
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)


# Guardar el modelo y el scaler juntos en un diccionario
modelo = {
    "scaler": scaler,
    "clf": clf
}

# Serializar con pickle
with open("svm_digits_model.pkl", "wb") as f:
    pickle.dump(modelo, f)

# Estilos personalizados
st.markdown(
    """
    <style>
    .big-header {
        font-size: 36px;
        font-weight: bold;
        color: #4B0082;
        text-align: left;
        margin-top: 20px;
    }
    .section-title {
        font-size: 24px;
        color: #006400;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .code-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-left: 4px solid #4B0082;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<div class='big-header'>Modelo guardado como svm_digits_model.pkl</div>", unsafe_allow_html=True)

