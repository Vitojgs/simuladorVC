import streamlit as st
import cv2
import numpy as np
import math
import tempfile
from PIL import Image
import os

# Configuracao da Pagina
st.set_page_config(page_title="Simulador de Visao - CitrID", layout="wide")

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
    type=['jpg', 'png', 'jpeg', 'avi']
)

# --- NOVO BLOCO: GUIA DE UTILIZAÇÃO ---
st.sidebar.markdown("---")
with st.sidebar.expander("📚 Guia de Utilização"):
    st.markdown("""
    **Como utilizar esta ferramenta:**
    
    **1. Upload de Ficheiro e Extração de Frames**
    * Para calibrar os algoritmos, necessita de **carregar uma imagem estática**.
    * Se apenas tiver o vídeo (`.avi`), carregue-o e vá ao separador **4. Filtro de Relevância (Vídeo)**.
    * Clique em **"Iniciar Processamento"** e escolha um dos *frames* gerados.
    * Clique com o botão direito do rato sobre a imagem, selecione **"Guardar imagem como..."** e guarde-a no seu equipamento.
    * Remova o vídeo da barra lateral, faça o upload dessa nova imagem e inicie o seu estudo.
    
    **2. Segmentação HSV**
    * Utilize os *sliders* para encontrar os valores exatos que isolam a cor laranja. 
    * O objetivo é que a "Máscara Binária" apresente as laranjas a branco e o fundo a preto.
    * *Dica:* Copie a sugestão de código gerada no fundo da página para o seu ficheiro C++.
    
    **3. Operações Morfológicas**
    * Limpe o ruído (pontos brancos falsos) ou tape os buracos dentro das laranjas utilizando as ferramentas de Erosão, Dilação, Abertura ou Fecho.
    
    **4. Análise de Blobs**
    * Ajuste o Filtro de Área mínima para ignorar poeiras ou falsos positivos.
    * Ajuste a Tolerância de Circularidade para afinar a precisão matemática entre as Categorias (Extra, I e II).
    
    **5. Filtro de Relevância (Vídeo)**
    * Exclusivo para vídeo. Permite ignorar *frames* vazios no tapete rolante e extrair apenas os momentos com fruta visível para análise estática.
    """)
# --------------------------------------

# Constantes do Projeto
PIXEL_TO_MM = 55.0 / 280.0

def carregar_imagem(file):
    image = Image.open(file)
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Streamlit (RGB) para OpenCV (BGR)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array, img_bgr
    return None, None

# Logica de Processamento
if uploaded_file is None:
    st.title("Simulador de Calibracao CitrID")
    st.info("Utilize a barra lateral para carregar um ficheiro de imagem ou video para iniciar.")
else:
    # Verificacao do tipo de ficheiro
    is_video = uploaded_file.name.lower().endswith('.avi')

    if not is_video:
        img_rgb, img_bgr = carregar_imagem(uploaded_file)
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. Segmentacao HSV
    if menu == "1. Segmentacao HSV":
        if is_video:
            st.warning("Por favor, carregue uma imagem estatica para calibrar os limites HSV.")
        else:
            st.title("Simulador de Segmentacao HSV")
            st.markdown("Ajuste os limites para isolar a cor das laranjas.")

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
            
            colA, colB = st.columns(2)
            colA.image(img_rgb, caption="Imagem Original", use_container_width=True)
            colB.image(mask, caption="Mascara Binaria", use_container_width=True)

            st.subheader("Sugestao de Codigo C++")
            st.code(f"if (h >= {(h_min/179.0)*360.0:.1f} && h <= {(h_max/179.0)*360.0:.1f} && s >= {s_min/255.0:.2f} && v >= {v_min/255.0:.2f})", language="cpp")

    # 2. Operacoes Morfologicas
    elif menu == "2. Operacoes Morfologicas":
        if is_video:
            st.warning("Esta funcionalidade requer o upload de uma imagem.")
        else:
            st.title("Simulador de Operacoes Morfologicas")
            st.markdown("Limpeza de ruido e preenchimento de falhas na mascara.")

            mask_base = cv2.inRange(hsv_img, np.array([8, 100, 75]), np.array([35, 255, 255]))

            col1, col2 = st.columns(2)
            with col1:
                k_size = st.slider("Tamanho do Kernel (Impar)", 1, 31, 5, step=2)
            with col2:
                op_tipo = st.selectbox("Operacao", ["Fecho (Close)", "Abertura (Open)", "Dilatacao", "Erosao"])

            kernel = np.ones((k_size, k_size), np.uint8)

            if op_tipo == "Dilatacao":
                resultado = cv2.dilate(mask_base, kernel, iterations=1)
            elif op_tipo == "Erosao":
                resultado = cv2.erode(mask_base, kernel, iterations=1)
            elif op_tipo == "Fecho (Close)":
                resultado = cv2.morphologyEx(mask_base, cv2.MORPH_CLOSE, kernel)
            else:
                resultado = cv2.morphologyEx(mask_base, cv2.MORPH_OPEN, kernel)

            colA, colB = st.columns(2)
            colA.image(mask_base, caption="Mascara Original", use_container_width=True)
            colB.image(resultado, caption=f"Resultado: {op_tipo}", use_container_width=True)

    # 3. Analise de Blobs
    elif menu == "3. Analise de Blobs":
        if is_video:
            st.warning("Esta funcionalidade requer o upload de uma imagem.")
        else:
            st.title("Extracao de Blobs e Categorias")
            
            mask = cv2.inRange(hsv_img, np.array([8, 100, 75]), np.array([35, 255, 255]))
            kernel = np.ones((5, 5), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            col1, col2 = st.columns(2)
            with col1:
                area_min = st.slider("Filtro de Area Minima (px)", 100, 5000, 1500)
            with col2:
                tol_extra = st.slider("Circularidade Minima Categoria EXTRA", 0.50, 1.00, 0.88, step=0.01)

            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            img_resultado = img_rgb.copy()
            laranjas_count = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < area_min:
                    continue
                    
                laranjas_count += 1
                perimetro = cv2.arcLength(cnt, True)
                x, y, w, h = cv2.boundingRect(cnt)

                if perimetro == 0: continue
                circularidade = (4 * math.pi * area) / (perimetro ** 2)

                if circularidade >= tol_extra:
                    cat, cor = "EXTRA", (0, 255, 0)
                elif circularidade >= (tol_extra - 0.10):
                    cat, cor = "CAT I", (255, 255, 0)
                else:
                    cat, cor = "CAT II", (255, 0, 0)

                cv2.rectangle(img_resultado, (x, y), (x+w, y+h), cor, 2)
                cv2.putText(img_resultado, f"{cat} ({circularidade:.2f})", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

            st.write(f"Laranjas detectadas: {laranjas_count}")
            st.image(img_resultado, caption="Analise de Blobs", use_container_width=True)

    # 4. Filtro de Relevancia (Video)
    elif menu == "4. Filtro de Relevancia (Video)":
        st.title("Analisador de Relevancia de Video")
        
        if is_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("Threshold de Densidade (%)", 0.0, 10.0, 0.5, 0.1)
            with col2:
                skip_frames = st.number_input("Analisar a cada N frames", min_value=1, value=15)

            if st.button("Iniciar Processamento"):
                selected_frames = []
                frame_idx = 0
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                bar = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if frame_idx % skip_frames == 0:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, np.array([8, 100, 75]), np.array([35, 255, 255]))
                        density = (cv2.countNonZero(mask) / mask.size) * 100
                        
                        if density >= threshold:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            selected_frames.append((frame_idx, density, frame_rgb))
                    
                    frame_idx += 1
                    if total_f > 0: bar.progress(min(frame_idx/total_f, 1.0))
                
                cap.release()
                st.success(f"Analise terminada. Encontrados {len(selected_frames)} momentos de interesse.")

                for idx, dens, img in selected_frames:
                    with st.expander(f"Frame {idx} - Densidade: {dens:.2f}%"):
                        st.image(img, use_container_width=True)
            
            # Limpeza do ficheiro temporario
            tfile.close()
            if os.path.exists(tfile.name):
                os.remove(tfile.name)
        else:
            st.warning("Por favor, carregue um ficheiro .avi para esta analise.")
