import streamlit as st
import google.generativeai as genai
from PIL import Image  # Librería para manejar imágenes en Python
import numpy as np
import cv2 # Librería para Procesamiento Imagenes y Videos
import tempfile #Librería para crear archivos temporales en disco
import os # Importa módulo para operaciones de archivos (sistema operativo)
import time # Para sleep quota retry

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
    ('Imagen (Contador)', 'Audio (Transcripción)', 'Video (Análisis)')
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

      image_array = np.array(image)
      with st.expander("📊 Datos Técnicos de la Matriz (Señal Visual)"):
        st.write(f"Dimensiones de la matriz (Píxeles): {image_array.shape}")
        st.write(f"Valor Máximo de Intensidad: {image_array.max()}")
        st.write(f"Valor Mínimo de Intensidad: {image_array.min()}")
      
      
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
        # Datos del audio
        with st.expander("📊 Datos de la Señal de Entrada del Audio"):
            st.write(f"Formato: {uploaded_audio.type}")
            st.write(f"Tamaño: {uploaded_audio.size / 1024:.2f} KB")

        
        #col1, col2 = st.columns(2)
        #with col1:
        #    st.metric("Formato", uploaded_audio.type)
        #with col2:
        #    st.metric("Tamaño", f"{uploaded_audio.size / 1024:.2f} KB")
            
        # Explicación técnica para el taller:
        st.caption("Nota: La señal se digitaliza y se envía como un flujo de bytes codificados en Base64 hacia los tensores del modelo Gemini.")

        
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

# --- LÓGICA PARA VIDEO (Lo nuevo) ---
# --- VIDEO 2 FRAMES (PRIMERO + ÚLTIMO) ---
elif option == 'Video (Análisis)':
    st.write("🎥 Análisis Smart: Frame 1 + Último (quota-safe)")
    
    uploaded_video = st.file_uploader("Sube video...", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎬 Analizar 2 Frames", type="primary"):
                st.session_state.video_process = True
                st.rerun()
        
        if st.session_state.get('video_process', False):
            st.info("🚀 Procesando 2 frames (1min espera quota)...")
            
            # ARCHIVO TEMPORAL
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            results = []
            
            # FRAME 1: PRIMERO
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.info("📱 Frame 1/2: INICIO video")
                response = model.generate_content(["Describe esta escena (inicio video).", image])
                results.append(f"⏰ **INICIO** (0s): {response.text}")
                st.success("✅ Frame 1 OK")
            
            # ESPERA QUOTA
            st.info("⏳ Esperando 60s (quota Gemini)...")
            time.sleep(60)
            
            # FRAME 2: ÚLTIMO  
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = cap.read()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.info("🏁 Frame 2/2: FINAL video")
                response = model.generate_content(["Describe esta escena (final video).", image])
                results.append(f"⏰ **FINAL** ({(total_frames-1)/fps:.1f}s): {response.text}")
                st.success("✅ Frame 2 OK")
            
            # LIMPIEZA
            cap.release()
            import os
            os.unlink(tfile.name)
            
            # RESULTADOS
            st.session_state.video_process = False
            st.balloons()
            st.markdown("---")
            st.subheader("🎬 RESUMEN VIDEO:")
            for result in results:
                st.markdown(f"• {result}")
            
            st.button("🔄 Nuevo Video")



else:
    st.info("👆 Por favor, sube una imagen para comenzar.")
