import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pickle




#Este es de refuerzo, pero no lo uso
def preprocess_canvas_image(image_data):
    # Convertir a escala de grises
    image = Image.fromarray(image_data.astype('uint8')).convert("L")

    # Convertir a numpy array
    image_np = np.array(image)

    # Recortar el contenido blanco (dígito)
    _, thresh = cv2.threshold(image_np, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return np.zeros((1, 64))  # Imagen vacía

    x, y, w, h = cv2.boundingRect(coords)
    cropped = image_np[y:y+h, x:x+w]

    # Redimensionar a 8x8 píxeles (como en el dataset)
    resized = cv2.resize(cropped, (8, 8), interpolation=cv2.INTER_AREA)

    # Escalar valores al rango 0-16
    scaled = (resized / 255.0) * 16.0

    # Aplanar y normalizar con el scaler original
    flat = scaled.flatten().reshape(1, -1)
    flat = scaler.transform(flat)

    return flat


def preprocesar_canvas_para_svm_1(image_data, scaler):
    """
    Prepara la imagen del canvas para que sea compatible con un modelo SVM entrenado con digits (8x8).
    
    Parámetros:
        image_data: imagen en formato numpy.array de canvas.image_data
        scaler: el StandardScaler que usaste para entrenar el modelo

    Devuelve:
        La imagen transformada y lista para hacer predicción con clf.predict()
    """

    if image_data is None:
        return None

    # Convertir a escala de grises con PIL
    imagen_pil = Image.fromarray(image_data.astype("uint8")).convert("L")

    # Redimensionar a 8x8 (como el dataset digits)
    imagen_pil = imagen_pil.resize((8, 8))

    # Convertir a array y normalizar de 0-255 a 0-16 (como digits)
    imagen_np = np.array(imagen_pil)
    imagen_np = 16 * (imagen_np / 255.0)

    # Aplanar y escalar
    imagen_flat = imagen_np.flatten().reshape(1, -1)
    imagen_scaled = scaler.transform(imagen_flat)

    return imagen_scaled


def colored_header(texto, color="black", align="left", size=2):
    """
    Muestra un encabezado de Streamlit con color y alineación personalizada.

    Parámetros:
        texto: el texto a mostrar
        color: el color del texto (por nombre o en hexadecimal)
        align: alineación ('left', 'center', 'right')
        size: nivel de encabezado (1 a 6, donde 1 es el más grande)
    """
    st.markdown(
        f"<h{size} style='color:{color}; text-align:{align};'>{texto}</h{size}>",
        unsafe_allow_html=True
    )


# Cargar el modelo desde el archivo
with open("svm_digits_model.pkl", "rb") as f:
    modelo = pickle.load(f)

scaler = modelo["scaler"]
clf = modelo["clf"]

# Create a canvas to draw the digit
canvas = st_canvas(
    fill_color="rgb(0, 0, 0)",
    stroke_width=20,
    stroke_color="rgb(255, 255, 255)",
    background_color="rgb(0, 0, 0)",
    width=150,
    height=150,
    drawing_mode="freedraw",
    key="canvas",
)

colored_header("Detección de Dígitos con OpenCV y Scikit-Learn", color="purple", size=4)
st.markdown("<h5 style='color: green;'>Dibuja un número en el lienzo y deja que el modelo lo prediga.</h5>", unsafe_allow_html=True)




def preprocess_image(image):
    # Convertir a escala de grises
    image = image.convert("L")
    #Redimensionar, ya que SVM fue entrenado con imagenes 8x8
    image = image.resize((8, 8))
    
    #Necesitamos convertirla en una matriz de números, y que 
    # sean hexadecimales, no de 0 a 255, porque digits usa eso, por eso escalamos
    image_array = np.array(image)
    image_array = 16 * (image_array / 255.0)

    # Aplanar la imagen para que sea un vector de 64 elementos
    image_array = image_array.flatten().reshape(1, -1)

    # Aplicar el mismo scaler que usaste al entrenar
    image_array = scaler.transform(image_array)

    # SVM expera (n_samples, n_features)
    # Aseguramos una fila (1 imagen), con 64 columnas (64 píxeles).
    #image_array = image_array.reshape(1, -1)  
    return image_array




def preprocesar_canvas_para_svm(image_data, scaler):
    if image_data is None:
        return None

    imagen_scaled = preprocess_image(Image.fromarray(image_data.astype("uint8")))
    return imagen_scaled



#Predice mediante la imagen 
def predict(image):
    image_array = preprocess_image(image)
    prediccion = clf.predict(image_array)
    return prediccion[0]



# Predice con la imagen
if st.button("Predict"):
    if canvas.image_data is not None:
        img_processed = preprocesar_canvas_para_svm(canvas.image_data, scaler)
        prediction = clf.predict(img_processed)[0]
        st.subheader("Predicción")
        st.write(f"El modelo predice que el número es: **{prediction}**")



st.markdown("""
---
---

""")

colored_header("Subir una imagen para predicción", color="purple", size=4)
st.write("<h5 style='color: green;'>Selecciona un dígito del dataset o sube una imagen para predecir el número.</h5>", unsafe_allow_html=True)
archivo_subido = st.file_uploader("Sube una imagen manuscrita (JPG o PNG)", type=["jpg", "png"])

if archivo_subido is not None:
    # Mostrar imagen con PIL
    image = Image.open(archivo_subido)
    st.image(image, caption='Imagen subida',  width=150)  
    st.write("")

    # Make a prediction
    prediction = predict(image)
    #prediccion = predecir_digito(archivo_subido)
    st.subheader(f"✅ El modelo predice que el número es: **{prediction}**")

st.write("Esta app usa OpenCV para procesar imágenes y Scikit-learn para predecir dígitos manuscritos.")




#Este es de refuerzo, pero no lo uso
def preprocess_canvas_image(image_data):
    # Convertir a escala de grises
    image = Image.fromarray(image_data.astype('uint8')).convert("L")

    # Convertir a numpy array
    image_np = np.array(image)

    # Recortar el contenido blanco (dígito)
    _, thresh = cv2.threshold(image_np, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return np.zeros((1, 64))  # Imagen vacía

    x, y, w, h = cv2.boundingRect(coords)
    cropped = image_np[y:y+h, x:x+w]

    # Redimensionar a 8x8 píxeles (como en el dataset)
    resized = cv2.resize(cropped, (8, 8), interpolation=cv2.INTER_AREA)

    # Escalar valores al rango 0-16
    scaled = (resized / 255.0) * 16.0

    # Aplanar y normalizar con el scaler original
    flat = scaled.flatten().reshape(1, -1)
    flat = scaler.transform(flat)

    return flat


def preprocesar_canvas_para_svm_1(image_data, scaler):
    """
    Prepara la imagen del canvas para que sea compatible con un modelo SVM entrenado con digits (8x8).
    
    Parámetros:
        image_data: imagen en formato numpy.array de canvas.image_data
        scaler: el StandardScaler que usaste para entrenar el modelo

    Devuelve:
        La imagen transformada y lista para hacer predicción con clf.predict()
    """

    if image_data is None:
        return None

    # Convertir a escala de grises con PIL
    imagen_pil = Image.fromarray(image_data.astype("uint8")).convert("L")

    # Redimensionar a 8x8 (como el dataset digits)
    imagen_pil = imagen_pil.resize((8, 8))

    # Convertir a array y normalizar de 0-255 a 0-16 (como digits)
    imagen_np = np.array(imagen_pil)
    imagen_np = 16 * (imagen_np / 255.0)

    # Aplanar y escalar
    imagen_flat = imagen_np.flatten().reshape(1, -1)
    imagen_scaled = scaler.transform(imagen_flat)

    return imagen_scaled