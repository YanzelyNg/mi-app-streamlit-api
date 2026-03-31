import streamlit as st
import google.generativeai as genai
from PIL import Image  # Librería para manejar imágenes en Python

# --- 1. CONFIGURACIÓN INICIAL ---
# st.secrets['GOOGLE_API_KEY'] busca la llave que guardamos en el Setting de Streamlit Cloud.
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("⚠️ Falta configurar la GOOGLE_API_KEY en los Secrets de Streamlit.")
    st.stop() # Detiene la ejecución si no hay llave

st.set_page_config(page_title="IA a tu Servicio", page_icon="🔢")
st.title("🔢 Contador de Objetos con IA")
st.write("Sube una foto y te contaré qué hay en ella.")

# --- 2. EL CARGADOR DE ARCHIVOS (Explicación para el taller) ---
# st.file_uploader crea el botón para subir archivos. Limitamos a imágenes.
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- 3. MOSTRAR LA IMAGEN EN LA APP ---
    # Convertimos el archivo subido en un objeto de imagen de Python (PIL)
    image = Image.open(uploaded_file)
    # st.image la muestra en la pantalla de la app
    st.image(image1, caption='Imagen cargada', use_column_width=True)
    
    # --- 4. EL BOTÓN DE ACCIÓN ---
    if st.button("Contar Objetos"):
        with st.spinner("Analizando la imagen..."):
            try:
                # --- 5. CONECTANDO CON GEMINI ---
                # Usamos gemini-1.5-flash porque es el más rápido para visión artificial.
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                # Este es el 'PROMPT': la instrucción específica para la IA.
                prompt = """
                Analiza esta imagen detalladamente. Tu tarea es identificar y contar los objetos principales.
                Por ejemplo, si ves frutas, di: 'Hay 3 manzanas, 2 plátanos y 1 naranja'.
                Sé preciso y numera la lista si hay varios tipos de objetos.
                """
                
                # --- 6. ENVIANDO DATOS A LA API ---
                # Enviamos una lista que contiene el texto (prompt) y la imagen.
                response = model.generate_content([prompt, image])
                
                # --- 7. MOSTRAR EL RESULTADO ---
                st.subheader("Resultado del conteo:")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("👆 Por favor, sube una imagen para comenzar.")
