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
    type=['jpg', 'png', 'jpeg', 'avi']
)

# Guia de Utilizacao
st.sidebar.markdown("---")
with st.sidebar.expander("📚 Guia de Utilização"):
    st.markdown("""
    **Como utilizar esta ferramenta:**

    **1. Upload de Ficheiro e Extração de Frames**
    * Para calibrar os algoritmos, necessita de **carregar uma imagem estática**.
    * Se apenas tiver o vídeo (`.avi`), carregue-o e vá ao separador **4. Filtro de Relevância (Vídeo)**.
    * Clique em **"Iniciar Processamento"** e escolha um dos *frames* gerados.
    * Utilize o botão **Descarregar em .PPM** para guardar o frame sem perdas.
    * Remova o vídeo da barra lateral, faça o upload dessa nova imagem e inicie o seu estudo.

    **2. Segmentação HSV**
    * Utilize os sliders para encontrar os valores exatos que isolam a cor laranja.
    * O objetivo é que a máscara binária apresente apenas as laranjas a branco e o fundo a preto.

    **3. Operações Morfológicas**
    * Limpe ruído e preencha falhas nas máscaras.

    **4. Análise de Blobs**
    * Ajuste área mínima e circularidade para melhorar a deteção.

    **5. Filtro de Relevância (Vídeo)**
    * Permite extrair automaticamente apenas os frames relevantes do vídeo.
    """)

PIXEL_TO_MM = 55.0 / 280.0


def carregar_imagem(file):
    image = Image.open(file)
    img_array = np.array(image)

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array, img_bgr

    return None, None


if uploaded_file is None:
    st.title("Simulador de Calibracao CitrID")
    st.info("Utilize a barra lateral para carregar um ficheiro de imagem ou vídeo para iniciar.")

else:
    is_video = uploaded_file.name.lower().endswith('.avi')

    if not is_video:
        img_rgb, img_bgr = carregar_imagem(uploaded_file)
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 4. Filtro de Relevancia (Video)
    if menu == "4. Filtro de Relevancia (Video)":
        st.title("Analisador de Relevancia de Video")

        if is_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            col1, col2 = st.columns(2)

            with col1:
                threshold = st.slider(
                    "Threshold de Densidade (%)",
                    0.0,
                    10.0,
                    0.5,
                    0.1
                )

            with col2:
                skip_frames = st.number_input(
                    "Analisar a cada N frames",
                    min_value=1,
                    value=15
                )

            if st.button("Iniciar Processamento"):
                selected_frames = []
                frame_idx = 0
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                bar = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % skip_frames == 0:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                        mask = cv2.inRange(
                            hsv,
                            np.array([5, 140, 80]),
                            np.array([22, 255, 255])
                        )

                        density = (cv2.countNonZero(mask) / mask.size) * 100

                        if density >= threshold:
                            nome_ficheiro = f"frames_ppm/frame_{frame_idx}.ppm"
                            cv2.imwrite(nome_ficheiro, frame)

                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            selected_frames.append(
                                (frame_idx, density, frame_rgb, nome_ficheiro)
                            )

                    frame_idx += 1

                    if total_f > 0:
                        bar.progress(min(frame_idx / total_f, 1.0))

                cap.release()

                st.success(
                    f"Análise terminada. Encontrados {len(selected_frames)} momentos de interesse."
                )

                for idx, dens, img, ficheiro in selected_frames:
                    with st.expander(
                        f"Frame {idx} - Densidade: {dens:.2f}%"
                    ):
                        st.image(img, use_container_width=True)

                        with open(ficheiro, "rb") as file:
                            st.download_button(
                                label=f"Descarregar Frame {idx} em .PPM",
                                data=file,
                                file_name=os.path.basename(ficheiro),
                                mime="image/x-portable-pixmap"
                            )

            tfile.close()
            if os.path.exists(tfile.name):
                os.remove(tfile.name)

        else:
            st.warning("Por favor, carregue um ficheiro .avi para esta análise.")
