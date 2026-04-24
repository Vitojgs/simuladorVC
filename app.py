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
    "Carregue uma imagem ou video (.avi, .ppm)",
    type=['jpg', 'png', 'jpeg', 'ppm', 'avi']
)

# Guia de Utilizacao
with st.sidebar.expander("📚 Guia de Utilização"):
    st.markdown("""
    1. **Upload:** Carregue imagem para calibrar ou vídeo para extrair frames.
    2. **Segmentação:** Ajuste limites HSV para isolar as laranjas.
    3. **Morfologia:** Limpeza de ruído e preenchimento de falhas.
    4. **Blobs:** Análise de área e circularidade para classificação.
    5. **Vídeo:** Extração automática de frames relevantes para estudo estático.
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
        colB.image(mask, caption="Máscara Binária", use_container_width=True)
        st.code(f"if (h >= {(h_min/179.0)*360.0:.1f} && h <= {(h_max/179.0)*360.0:.1f} && s >= {s_min/255.0:.2f} && v >= {v_min/255.0:.2f})", language="cpp")

# --- MENU: OPERAÇÕES MORFOLÓGICAS ---
elif menu == "2. Operacoes Morfologicas":
    if hsv_img is not None:
        st.title("Operacoes Morfologicas")
        mask_base = cv2.inRange(hsv_img, np.array([5, 140, 80]), np.array([22, 255, 255]))
        k_size = st.slider("Tamanho do Kernel", 3, 21, 5, step=2)
        op = st.selectbox("Operação", ["Fecho", "Abertura", "Dilatação", "Erosão"])
        kernel = np.ones((k_size, k_size), np.uint8)
        
        if op == "Fecho": res = cv2.morphologyEx(mask_base, cv2.MORPH_CLOSE, kernel)
        elif op == "Abertura": res = cv2.morphologyEx(mask_base, cv2.MORPH_OPEN, kernel)
        else: res = cv2.dilate(mask_base, kernel) if op == "Dilatação" else cv2.erode(mask_base, kernel)
        
        st.image(res, caption=op, use_container_width=True)

# --- MENU: ANÁLISE DE BLOBS ---
elif menu == "3. Analise de Blobs":
    if hsv_img is not None:
        st.title("Analise de Blobs")
        mask = cv2.inRange(hsv_img, np.array([5, 140, 80]), np.array([22, 255, 255]))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = img_rgb.copy()
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                count += 1
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(res, (x,y), (x+w, y+h), (0,255,0), 2)
        st.write(f"Detectadas: {count}")
        st.image(res, caption="Blobs Identificados", use_container_width=True)

# --- MENU: FILTRO DE RELEVÂNCIA ---
elif menu == "4. Filtro de Relevancia (Video)":
    st.title("Analisador de Relevancia de Video")
    if is_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        if st.button("Processar"):
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
        st.warning("Carregue um vídeo .avi para esta funcionalidade.")
