import streamlit as st
import cv2
import numpy as np
import math
import tempfile
from PIL import Image
import os
import zipfile
import io

# =====================================================
# CONFIGURAÇÃO INICIAL
# =====================================================

st.set_page_config(
    page_title="Simulador de Visão - CitrID",
    layout="wide"
)

# Pasta para guardar frames extraídos
os.makedirs("frames_ppm", exist_ok=True)

# Persistência entre reruns
if "selected_frames" not in st.session_state:
    st.session_state.selected_frames = []

# Valores HSV recomendados para laranja
HSV_LOWER = np.array([5, 140, 80])
HSV_UPPER = np.array([22, 255, 255])


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================


def carregar_imagem(file):
    """Carrega imagem de forma robusta (incluindo .ppm)."""
    try:
        image = Image.open(file).convert("RGB")
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_rgb, img_bgr
    except Exception as e:
        st.error(f"Erro ao abrir imagem: {e}")
        return None, None


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Configuração e Navegação")

menu = st.sidebar.radio(
    "Escolha o Simulador:",
    (
        "1. Segmentação HSV",
        "2. Operações Morfológicas",
        "3. Análise de Blobs",
        "4. Filtro de Relevância (Vídeo)",
    ),
)

st.sidebar.markdown("---")
st.sidebar.header("Upload de Ficheiro")

uploaded_file = st.sidebar.file_uploader(
    "Carregue uma imagem ou vídeo (.avi, .ppm)",
    type=["jpg", "jpeg", "png", "ppm", "avi"],
)

st.sidebar.markdown("---")
with st.sidebar.expander("📚 Guia de Utilização"):
    st.markdown(
        """
### Como utilizar

**1. Upload**
- Carregue uma imagem para calibrar HSV
- Ou carregue um vídeo `.avi` para extrair frames

**2. Segmentação HSV**
- Ajuste os sliders até obter apenas as laranjas a branco

**3. Operações Morfológicas**
- Limpe ruído e preencha falhas

**4. Análise de Blobs**
- Ajuste área mínima e circularidade
- Classificação: EXTRA / CAT I / CAT II

**5. Filtro de Relevância**
- Extrai apenas frames com fruta visível
- Permite download individual ou ZIP completo
        """
    )


# =====================================================
# PROCESSAMENTO INICIAL
# =====================================================

img_rgb = None
img_bgr = None
hsv_img = None
is_video = False

if uploaded_file is not None:
    is_video = uploaded_file.name.lower().endswith(".avi")

    if not is_video:
        img_rgb, img_bgr = carregar_imagem(uploaded_file)
        if img_bgr is not None:
            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


# =====================================================
# MENU 1 - SEGMENTAÇÃO HSV
# =====================================================

if menu == "1. Segmentação HSV":
    st.title("Simulador de Segmentação HSV")

    if uploaded_file is None:
        st.info("Carregue uma imagem para iniciar.")

    elif is_video:
        st.warning("Carregue uma imagem estática para calibrar HSV.")

    elif hsv_img is not None:
        st.markdown("Ajuste os limites para isolar a cor das laranjas.")

        c1, c2, c3 = st.columns(3)

        with c1:
            h_min = st.slider("Hue Min", 0, 179, 5)
            h_max = st.slider("Hue Max", 0, 179, 22)

        with c2:
            s_min = st.slider("Saturation Min", 0, 255, 140)
            s_max = st.slider("Saturation Max", 0, 255, 255)

        with c3:
            v_min = st.slider("Value Min", 0, 255, 80)
            v_max = st.slider("Value Max", 0, 255, 255)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv_img, lower, upper)

        col_a, col_b = st.columns(2)
        col_a.image(img_rgb, caption="Imagem Original", use_container_width=True)
        col_b.image(mask, caption="Máscara Binária", use_container_width=True)

        st.subheader("Sugestão de Código C++")
        st.code(
            f"if (h >= {(h_min/179.0)*360.0:.1f} && "
            f"h <= {(h_max/179.0)*360.0:.1f} && "
            f"s >= {s_min/255.0:.2f} && "
            f"v >= {v_min/255.0:.2f})",
            language="cpp",
        )


# =====================================================
# MENU 2 - OPERAÇÕES MORFOLÓGICAS
# =====================================================

elif menu == "2. Operações Morfológicas":
    st.title("Operações Morfológicas")

    if hsv_img is None or is_video:
        st.warning("Esta funcionalidade requer uma imagem.")

    else:
        mask_base = cv2.inRange(hsv_img, HSV_LOWER, HSV_UPPER)

        c1, c2 = st.columns(2)

        with c1:
            k_size = st.slider(
                "Tamanho do Kernel (ímpar)",
                3,
                31,
                5,
                step=2,
            )

        with c2:
            operacao = st.selectbox(
                "Operação",
                ["Fecho", "Abertura", "Dilatação", "Erosão"],
            )

        kernel = np.ones((k_size, k_size), np.uint8)

        if operacao == "Fecho":
            resultado = cv2.morphologyEx(mask_base, cv2.MORPH_CLOSE, kernel)
        elif operacao == "Abertura":
            resultado = cv2.morphologyEx(mask_base, cv2.MORPH_OPEN, kernel)
        elif operacao == "Dilatação":
            resultado = cv2.dilate(mask_base, kernel, iterations=1)
        else:
            resultado = cv2.erode(mask_base, kernel, iterations=1)

        col_a, col_b = st.columns(2)
        col_a.image(mask_base, caption="Máscara Original", use_container_width=True)
        col_b.image(resultado, caption=operacao, use_container_width=True)


# =====================================================
# MENU 3 - ANÁLISE DE BLOBS
# =====================================================

elif menu == "3. Análise de Blobs":
    st.title("Análise de Blobs e Classificação")

    if hsv_img is None or is_video:
        st.warning("Esta funcionalidade requer uma imagem.")

    else:
        mask = cv2.inRange(hsv_img, HSV_LOWER, HSV_UPPER)
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        c1, c2 = st.columns(2)

        with c1:
            area_min = st.slider(
                "Área mínima (px)",
                500,
                8000,
                3000,
            )

        with c2:
            tol_extra = st.slider(
                "Circularidade mínima EXTRA",
                0.50,
                1.00,
                0.88,
                step=0.01,
            )

        contours, _ = cv2.findContours(
            mask_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        resultado = img_rgb.copy()
        total = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_min:
                continue

            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Filtro por proporção (quase circular)
            ratio = w / h
            if ratio < 0.85 or ratio > 1.15:
                continue

            circularidade = (4 * math.pi * area) / (perimetro ** 2)
            total += 1

            if circularidade >= tol_extra:
                categoria = "EXTRA"
                cor = (0, 255, 0)
            elif circularidade >= (tol_extra - 0.10):
                categoria = "CAT I"
                cor = (255, 255, 0)
            else:
                categoria = "CAT II"
                cor = (255, 0, 0)

            cv2.rectangle(resultado, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(
                resultado,
                f"{categoria} ({circularidade:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                cor,
                2,
            )

        st.write(f"Laranjas detetadas: {total}")
        st.image(resultado, caption="Resultado da Análise", use_container_width=True)


# =====================================================
# MENU 4 - FILTRO DE RELEVÂNCIA (VÍDEO)
# =====================================================

elif menu == "4. Filtro de Relevância (Vídeo)":
    st.title("Filtro de Relevância de Vídeo")

    if uploaded_file is None:
        st.info("Carregue um ficheiro .avi para iniciar.")

    elif not is_video:
        st.warning("Esta funcionalidade requer um vídeo .avi.")

    else:
        c1, c2 = st.columns(2)

        with c1:
            threshold = st.slider(
                "Threshold de Densidade (%)",
                0.0,
                10.0,
                0.5,
                0.1,
            )

        with c2:
            skip_frames = st.number_input(
                "Analisar a cada N frames",
                min_value=1,
                value=15,
            )

        if st.button("Iniciar Processamento"):
            st.session_state.selected_frames = []

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            progress = st.progress(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % skip_frames == 0:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

                    density = (cv2.countNonZero(mask) / mask.size) * 100

                    if density >= threshold:
                        nome = f"frames_ppm/frame_{frame_idx}.ppm"
                        cv2.imwrite(nome, frame)

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        st.session_state.selected_frames.append(
                            (frame_idx, density, frame_rgb, nome)
                        )

                frame_idx += 1

                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0))

            cap.release()

            if os.path.exists(tfile.name):
                os.remove(tfile.name)

        if st.session_state.selected_frames:
            st.success(
                f"Encontrados {len(st.session_state.selected_frames)} frames relevantes."
            )

            # ZIP com todos os frames
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for _, _, _, ficheiro in st.session_state.selected_frames:
                    zip_file.write(ficheiro, os.path.basename(ficheiro))

            zip_buffer.seek(0)

            st.download_button(
                label="Descarregar Todos os Frames (.ZIP)",
                data=zip_buffer,
                file_name="frames_ppm.zip",
                mime="application/zip",
            )

            # Download individual
            for idx, dens, img, ficheiro in st.session_state.selected_frames:
                with st.expander(
                    f"Frame {idx} - Densidade: {dens:.2f}%"
                ):
                    st.image(img, use_container_width=True)

                    with open(ficheiro, "rb") as file:
                        st.download_button(
                            label=f"Descarregar Frame {idx} em .PPM",
                            data=file,
                            file_name=os.path.basename(ficheiro),
                            mime="image/x-portable-pixmap",
                            key=f"download_{idx}",
                        )
