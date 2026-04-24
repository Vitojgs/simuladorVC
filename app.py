import streamlit as st
import cv2
import numpy as np
import math
import tempfile
from PIL import Image
import os

# Configuracao da Pagina
st.set_page_config(page_title="Simulador de Visao - CitrID", layout="wide")

# Criar pasta para guardar frames em .ppm
os.makedirs("frames_ppm", exist_ok=True)

# Barra Lateral - Navegacao
st.sidebar.title("Configuracao e Navegacao")
menu = st.sidebar.radio(
    "Escolha o Simulador:",
    (
        "1. Segmentacao HSV",
        "2. Operacoes Morfologicas",
        "3. Analise de Blobs",
        "4. Filtro de Relevancia (Video)"
    )
)

st.sidebar.markdown("---")
st.sidebar.header("Upload de Ficheiro")
uploaded_file = st.sidebar.file_uploader(
    "Carregue uma imagem ou video (.avi)",
    type=['jpg', 'png', 'jpeg', 'ppm', 'avi']
)

# Guia de Utilizacao
with st.sidebar.expander("📚 Guia de Utilização"):
    st.markdown("""
    1. **Upload:** Carregue imagem para calibrar ou vídeo para extrair frames.
    2. **Segmentação:** Ajuste os limites HSV.
    3. **Morfologia:** Limpeza de ruído.
    4. **Blobs:** Análise de área e circularidade.
    5. **Vídeo:** Extração automática de frames relevantes.
    """)

# Funcao robusta de carregamento
def carregar_imagem(file):
    image = Image.open(file).convert('RGB')
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array, img_bgr

# Processamento inicial
img_rgb = None
hsv_img = None
is_video = False

if uploaded_file is not None:
    is_video = uploaded_file.name.lower().endswith('.avi')
    if not is_video:
        img_rgb, img_bgr = carregar_imagem(uploaded_file)
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# --- MENU: SEGMENTAÇÃO HSV ---
if menu == "1. Segmentacao HSV":
    if is_video:
        st.warning("Carregue uma imagem estática para calibrar.")
    elif hsv_img is not None:
        st.title("Simulador de Segmentacao HSV")
        c1, c2, c3 = st.columns(3)
        h_min = c1.slider("Hue Min", 0, 179, 5)
        h_max = c1.slider("Hue Max", 0, 179, 22)
        s_min = c2.slider("Sat Min", 0, 255, 140)
        v_min = c3.slider("Val Min", 0, 255, 80)
        
        mask = cv2.inRange(hsv_img, np.array([h_min, s_min, v_min]), np.array([h_max, 255, 255]))
        colA, colB = st.columns(2)
        colA.image(img_rgb, caption="Original", use_container_width=True)
        colB.image(mask, caption="Máscara", use_container_width=True)

# --- MENU: OPERAÇÕES MORFOLÓGICAS ---
elif menu == "2. Operacoes Morfologicas":
        if is_video:
            st.warning("Esta funcionalidade requer o upload de uma imagem.")
        elif hsv_img is not None:
            st.title("Simulador de Operacoes Morfologicas")
            
            # Máscara base (ajusta aqui os teus valores ótimos)
            mask_base = cv2.inRange(hsv_img, np.array([8, 140, 80]), np.array([25, 255, 255]))

            col1, col2, col3 = st.columns(3)
            with col1:
                k_size = st.slider("Tamanho do Kernel (Ímpar)", 3, 21, 5, step=2)
            with col2:
                iters = st.slider("Número de Iterações", 1, 5, 1)
            with col3:
                op_tipo = st.selectbox("Operação", ["Fecho (Close)", "Abertura (Open)", "Dilatação", "Erosão"])

            kernel = np.ones((k_size, k_size), np.uint8)

            # Execução com iterações variáveis
            if op_tipo == "Dilatação":
                resultado = cv2.dilate(mask_base, kernel, iterations=iters)
            elif op_tipo == "Erosão":
                resultado = cv2.erode(mask_base, kernel, iterations=iters)
            elif op_tipo == "Fecho (Close)":
                resultado = cv2.morphologyEx(mask_base, cv2.MORPH_CLOSE, kernel, iterations=iters)
            else:
                resultado = cv2.morphologyEx(mask_base, cv2.MORPH_OPEN, kernel, iterations=iters)

            colA, colB = st.columns(2)
            colA.image(mask_base, caption="Máscara Original", use_container_width=True)
            colB.image(resultado, caption=f"Resultado: {op_tipo} (k={k_size}, it={iters})", use_container_width=True)

# --- MENU: ANÁLISE DE BLOBS ---
elif menu == "3. Analise de Blobs":
        # ... (código existente de máscara e contornos) ...
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_min: continue

            # Centroide
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            
            # Retângulo Orientado (Mais preciso)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # Desenhar
            cv2.drawContours(img_resultado, [box], 0, (0, 255, 0), 2)
            cv2.circle(img_resultado, (cx, cy), 5, (255, 0, 0), -1)
            
            # Mostrar Centroide
            cv2.putText(img_resultado, f"({cx},{cy})", (cx+10, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# --- MENU: FILTRO DE RELEVÂNCIA ---
elif menu == "4. Filtro de Relevancia (Video)":
    st.title("Analisador de Relevancia de Video")
    if is_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        if st.button("Iniciar Processamento"):
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % 15 == 0:
                    nome = f"frames_ppm/frame_{frame_idx}.ppm"
                    cv2.imwrite(nome, frame)
                    with open(nome, "rb") as f:
                        st.download_button(f"Baixar Frame {frame_idx}", f, file_name=f"frame_{frame_idx}.ppm")
                frame_idx += 1
        tfile.close()
    else:
        st.warning("Carregue um vídeo .avi")
