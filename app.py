import streamlit as st
import google.generativeai as genai
from PIL import Image  # Librería para manejar imágenes en Python
import numpy as np
import cv2 # Librería para Procesamiento Imagenes y Videos
import tempfile #Librería para crear archivos temporales en disco
import os # Importa módulo para operaciones de archivos (sistema operativo)

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
elif option == 'Video (Análisis)':
    st.write("Sube un video y analizaré algunos frames (máx. 5).")

    # 1. CARGA DEL VIDEO
    uploaded_video = st.file_uploader("Sube un video...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # MOSTRAR VIDEO EN LA APP
        st.video(uploaded_video)

        # 2. CREAR ARCHIVO TEMPORAL EN DISCO (necesario para OpenCV)
        # delete=False mantiene el archivo hasta que lo borremos manualmente
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())  # Escribe bytes del video
        tfile.close()  # Cierra el handle del archivo

        # 3. ABRIR VIDEO CON OPENCV PARA LEER METADATOS
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames del video
        fps = cap.get(cv2.CAP_PROP_FPS)                         # Frames por segundo
        duration = total_frames / fps if fps > 0 else 0         # Duración en segundos
        
        # MOSTRAR INFO TÉCNICA DEL VIDEO
        with st.expander("📊 Datos del Video"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Frames", total_frames)
            col2.metric("FPS", f"{fps:.1f}")
            col3.metric("Duración", f"{duration:.1f}s")
        
        # 4. BOTÓN PARA CONFIGURAR ANÁLISIS
        if st.button("Analizar Video"):
            # 5. CÁLCULO AUTOMÁTICO PARA MÁXIMO 5 FRAMES (EVITA QUOTA GEMINI)
            max_frames = 5  # Límite fijo para free tier (5 req/min)
            interval = max(1, total_frames // max_frames)  # Cada cuántos frames analizar
            frames_to_analyze = min(max_frames, total_frames)  # Nunca más de 5
            
            # MOSTRAR PREVIEW ANTES DE PROCESAR
            st.info(f"🔄 Analizaré **{frames_to_analyze} frames** máximo (de {total_frames} totales). "
                   f"Intervalo automático: cada {interval} frames. "
                   f"**{frames_to_analyze} requests** a Gemini.")
            
            # 6. CONFIRMACIÓN DEL USUARIO
            if st.button(f"✅ Confirmar: Procesar {frames_to_analyze} frames (máx 5)", type="primary"):
                with st.spinner("Procesando frames..."):
                    try:
                        results = []           # Lista para guardar descripciones
                        frame_count = 0        # Contador global de frames
                        analyzed_count = 0     # Contador de frames ANALIZADOS
                        
                        # 7. LOOP PRINCIPAL: LEER FRAME POR FRAME
                        while cap.isOpened() and analyzed_count < max_frames:
                            ret, frame = cap.read()  # Lee frame (ret=True si OK)
                            if not ret:
                                break
                                
                            # ¿Es este frame para analizar? (cada 'interval' frames)
                            if frame_count % interval == 0 and analyzed_count < max_frames:
                                # CONVERTIR FRAME PARA GEMINI
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB
                                image = Image.fromarray(frame_rgb)                  # PIL Image
                                
                                # LLAMADA A GEMINI (1 request por frame)
                                prompt = "Describe brevemente qué hay en esta escena."
                                response = model.generate_content([prompt, image])
                                
                                # GUARDAR RESULTADO CON TIMESTAMP
                                results.append(f"Frame {frame_count} ({frame_count/fps:.1f}s): {response.text}")
                                
                                analyzed_count += 1
                                st.status_update(f"Procesado: {analyzed_count}/{max_frames}")
                            
                            frame_count += 1  # Siguiente frame
                        
                        # 8. LIMPIEZA DE MEMORIA Y DISCO
                        cap.release()  # Libera VideoCapture
                        
                        # BORRAR ARCHIVO TEMPORAL (IMPORTANTE en Cloud)
                        os.unlink(tfile.name)  # Elimina /tmp/tmpXYZ.mp4
                        
                        # 9. MOSTRAR RESULTADOS
                        st.success(f"✅ Análisis completado: {len(results)} frames procesados.")
                        st.subheader("Resultados del video:")
                        for r in results:
                            st.write(r)
                            
                    except Exception as e:
                        st.error(f"Error en video: {e}")
                        # LIMPIEZA EN CASO DE ERROR
                        cap.release()
                        import os
                        os.unlink(tfile.name)
        else:
            st.info("👆 Haz clic en 'Analizar Video' para preview.")

else:
    st.info("👆 Por favor, sube una imagen para comenzar.")
