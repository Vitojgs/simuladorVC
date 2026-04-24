import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

# Configuração da Página
st.set_page_config(page_title="Simulador de Visão - CitrID", layout="wide")

st.sidebar.title("⚙️ Navegação")
menu = st.sidebar.radio(
    "Escolhe o Simulador:",
    ("1. Segmentação HSV", "2. Operações Morfológicas", "3. Análise de Blobs")
)

st.sidebar.markdown("---")
st.sidebar.header("Upload de Imagem")
uploaded_file = st.sidebar.file_uploader("Carrega uma frame do vídeo (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

def carregar_imagem(file):
    image = Image.open(file)
    # Converter de PIL para formato OpenCV (Array Numpy em BGR)
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # OpenCV usa BGR por defeito, mas a imagem do Streamlit vem em RGB
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array, img_bgr # Retorna RGB para display e BGR para processamento
    return None, None

if uploaded_file is None:
    st.title("Simulador de Calibração CitrID")
    st.info("Por favor, faz o upload de uma imagem na barra lateral para começar a simulação.")
else:
    img_rgb, img_bgr = carregar_imagem(uploaded_file)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 1. SIMULADOR HSV
    # ==========================================
    if menu == "1. Segmentação HSV":
        st.title("Simulador de Segmentação HSV")
        st.markdown("Ajusta os limites para isolar as laranjas.")

        col1, col2, col3 = st.columns(3)
        with col1:
            h_min = st.slider("Hue Min", 0, 179, 8)
            h_max = st.slider("Hue Max", 0, 179, 35)
        with col2:
            s_min = st.slider("Saturation Min", 0, 255, 100)
            s_max = st.slider("Saturation Max", 0, 255, 255)
        with col3:
            v_min = st.slider("Value Min", 0, 255, 75)
            v_max = st.slider("Value Max", 0, 255, 255)

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        
        # Display
        colA, colB = st.columns(2)
        colA.image(img_rgb, caption="Imagem Original", use_container_width=True)
        colB.image(mask, caption="Máscara Binária (Branco = Detetado)", use_container_width=True, channels="GRAY")

        st.code(f"// Dica para o teu código C++ (Lembra-te que em C++ a escala é diferente!)\nif (h >= {(h_min/179.0)*360.0:.1f} && h <= {(h_max/179.0)*360.0:.1f} && s >= {s_min/255.0:.2f} && v >= {v_min/255.0:.2f})", language="cpp")

    # ==========================================
    # 2. SIMULADOR MORFOLÓGICO
    # ==========================================
    elif menu == "2. Operações Morfológicas":
        st.title("Simulador Morfológico")
        st.markdown("Limpa o ruído e tapa os buracos da máscara.")

        # Máscara estática baseada num valor default decente de laranja
        mask_base = cv2.inRange(hsv_img, np.array([8, 100, 75]), np.array([35, 255, 255]))

        col1, col2 = st.columns(2)
        with col1:
            k_size = st.slider("Tamanho do Kernel (Ímpar)", 1, 31, 5, step=2)
        with col2:
            op_tipo = st.selectbox("Operação", ["Dilação", "Erosão", "Fecho (Close)", "Abertura (Open)"])

        kernel = np.ones((k_size, k_size), np.uint8)

        if op_tipo == "Dilação":
            resultado = cv2.dilate(mask_base, kernel, iterations=1)
        elif op_tipo == "Erosão":
            resultado = cv2.erode(mask_base, kernel, iterations=1)
        elif op_tipo == "Fecho (Close)":
            resultado = cv2.morphologyEx(mask_base, cv2.MORPH_CLOSE, kernel)
        else:
            resultado = cv2.morphologyEx(mask_base, cv2.MORPH_OPEN, kernel)

        colA, colB = st.columns(2)
        colA.image(mask_base, caption="Antes (Com Ruído/Buracos)", use_container_width=True, channels="GRAY")
        colB.image(resultado, caption=f"Depois da {op_tipo} (Kernel: {k_size}x{k_size})", use_container_width=True, channels="GRAY")

    # ==========================================
    # 3. SIMULADOR DE BLOBS
    # ==========================================
    elif menu == "3. Análise de Blobs":
        st.title("Extração de Blobs e Categorias")
        
        # Pipeline interno invisível para gerar a máscara
        mask = cv2.inRange(hsv_img, np.array([8, 100, 75]), np.array([35, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        col1, col2 = st.columns(2)
        with col1:
            area_min = st.slider("Filtro de Ruído: Área Mínima", 100, 5000, 1500)
        with col2:
            tol_extra = st.slider("Circularidade Mín. p/ Categoria EXTRA", 0.50, 1.00, 0.88, step=0.01)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_resultado = img_rgb.copy()
        laranjas_encontradas = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_min:
                continue
                
            laranjas_encontradas += 1
            perimetro = cv2.arcLength(cnt, True)
            
            x, y, w, h = cv2.boundingRect(cnt)
            xc, yc = x + w//2, y + h//2

            if perimetro == 0: continue
            
            circularidade = (4 * math.pi * area) / (perimetro ** 2)

            if circularidade >= tol_extra:
                categoria = "EXTRA"
                cor = (0, 255, 0) # Verde
            elif circularidade >= (tol_extra - 0.10):
                categoria = "CAT I"
                cor = (255, 255, 0) # Amarelo
            else:
                categoria = "CAT II"
                cor = (255, 0, 0) # Vermelho (No RGB Array)

            cv2.rectangle(img_resultado, (x, y), (x+w, y+h), cor, 2)
            cv2.circle(img_resultado, (xc, yc), 4, (255, 255, 255), -1)
            
            cv2.putText(img_resultado, f"Circ: {circularidade:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
            cv2.putText(img_resultado, categoria, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        st.success(f"**Laranjas detetadas:** {laranjas_encontradas}")
        st.image(img_resultado, caption="Análise Final", use_container_width=True)