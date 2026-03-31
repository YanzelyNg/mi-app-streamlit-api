import streamlit as st
import google.generativeai as genai
from PIL import Image  # Librería para manejar imágenes en Python

# --- CONFIGURACIÓN INICIAL ---
# st.secrets['GOOGLE_API_KEY'] busca la llave que guardamos en el Setting de Streamlit Cloud.
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("⚠️ Falta configurar la GOOGLE_API_KEY en los Secrets de Streamlit.")
    st.stop() # Detiene la ejecución si no hay llave

st.set_page_config(page_title="IA a tu Servicio", page_icon="🔢")
st.title("🔢 Contador de Objetos o Transcripción con IA")


# Menú en la barra lateral para el taller
option = st.sidebar.selectbox(
    '¿Qué quieres procesar hoy?',
    ('Imagen (Contador)', 'Audio (Transcripción)')
)

# --- CONECTANDO CON GEMINI ---
model = genai.GenerativeModel("gemini-2.5-flash")

if option == 'Imagen (Contador)':
  st.write("Sube una foto y te contaré qué hay en ella.")
  # --- EL CARGADOR DE ARCHIVOS (Explicación para el taller) ---
  # st.file_uploader crea el botón para subir archivos. Limitamos a imágenes.
  uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
  
  if uploaded_file is not None:
      # ---  MOSTRAR LA IMAGEN EN LA APP ---
      # Convertimos el archivo subido en un objeto de imagen de Python (PIL)
      image = Image.open(uploaded_file)
      # st.image la muestra en la pantalla de la app
      st.image(image, caption='Imagen cargada', use_column_width=True)
      
      
      # ---  EL BOTÓN DE ACCIÓN ---
      if st.button("Contar Objetos"):
          with st.spinner("Analizando la imagen..."):
              try:
                  # --- 5. CONECTANDO CON GEMINI ---
                  # Usamos gemini-1.5-flash porque es el más rápido para visión artificial.
                  #model = genai.GenerativeModel("gemini-2.5-flash")
                  
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

# --- LÓGICA PARA AUDIO (Lo nuevo) ---
elif option == 'Audio (Transcripción)':
    uploaded_audio = st.file_uploader("Sube un audio corto 10 seg máximo", type=["mp3", "wav", "m4a"])
    
    if uploaded_audio:
        st.audio(uploaded_audio)
        
        if st.button("Escuchar y Transcribir"):
            with st.spinner("La IA está escuchando..."):
                try:
                    # LEER EL AUDIO: Convertimos el archivo de Streamlit a bytes
                    audio_bytes = uploaded_audio.read()
                    
                    # ENVIAR A GEMINI:
                    # Pasamos el prompt y un diccionario con los datos del audio
                    response = model.generate_content([
                        "Transcribe este audio textualmente y luego haz un resumen de 3 puntos clave.",
                        {"mime_type": "audio/mp3", "data": audio_bytes}
                    ])
                    
                    st.subheader("Transcripción y Resumen:")
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Error en el audio: {e}")

else:
    st.info("👆 Por favor, sube una imagen para comenzar.")
