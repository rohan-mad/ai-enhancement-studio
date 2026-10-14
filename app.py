
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from io import BytesIO
import tempfile
import subprocess
import hashlib

ROOT = Path(__file__).resolve().parent

# Locate the Real-ESRGAN executable
possible_folders = [
    ROOT / "realesrgan-ncnn-vulkan-20220424-windows",
    ROOT / "realesrgan"
]

REAL_DIR = next(
    (
        p for p in possible_folders
        if (p / "realesrgan-ncnn-vulkan.exe").is_file()
    ),
    None
)

CNN_PATH = ROOT / "image_enhancement_cnn.pth"

st.set_page_config(
    page_title="Image Enhancement Studio",
    page_icon=None,
    layout="wide"
)

# --------------------------------------------------
# DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: #101319;
    color: #E7EAF0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #F5F6F8;
    letter-spacing: -0.035em;
    font-weight: 600;
}

.hero {
    padding: 1.8rem 0 2rem 0;
    border-bottom: 1px solid #2B303A;
    margin-bottom: 2rem;
}

.eyebrow {
    color: #A5B4FC;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

.hero h1 {
    font-size: clamp(2rem, 4vw, 2.8rem);
    margin: 0 0 0.7rem 0;
    line-height: 1.2;
}

.hero p {
    color: #A7AFBD;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 680px;
    margin: 0;
}

.section-label {
    color: #A5B4FC;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.section-title {
    color: #F3F4F6;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.muted {
    color: #9CA5B4;
    font-size: 0.9rem;
    line-height: 1.6;
}

.image-heading {
    color: #E7EAF0;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
}

div.stButton > button {
    border-radius: 9px;
    min-height: 2.8rem;
    font-weight: 600;
    transition: border-color 0.15s ease;
}

div.stButton > button[kind="primary"] {
    background: #6366F1;
    color: white;
    border: 1px solid #6366F1;
}

div.stButton > button[kind="primary"]:hover {
    background: #5558E8;
    border-color: #5558E8;
}

div[data-testid="stFileUploader"] {
    background: #171B24;
    border: 1px solid #303744;
    border-radius: 12px;
    padding: 0.8rem;
}

div[data-testid="stMetric"] {
    background: #171B24;
    border: 1px solid #303744;
    border-radius: 10px;
    padding: 0.9rem;
}

div[data-testid="stMetricLabel"] {
    color: #A7AFBD;
}

hr {
    border-color: #2B303A;
}

.footer {
    border-top: 1px solid #2B303A;
    padding-top: 1.2rem;
    margin-top: 3rem;
    text-align: center;
    color: #818A99;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# OUR EXPERIMENTAL CNN
# --------------------------------------------------

class ImageEnhancementCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )

        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return (x + self.layers(x)).clamp(0, 1)


@st.cache_resource
def load_cnn():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CNN_PATH,
        map_location=device,
        weights_only=True
    )

    model = ImageEnhancementCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def tensor_to_image(tensor):
    array = (
        tensor[0].detach().cpu()
        .permute(1, 2, 0).numpy()
    )
    array = np.clip(array * 255, 0, 255).round().astype(np.uint8)
    return Image.fromarray(array)


def run_realesrgan(original):
    if REAL_DIR is None:
        raise FileNotFoundError(
            "Real-ESRGAN was not found. Check its folder next to app.py."
        )

    exe = REAL_DIR / "realesrgan-ncnn-vulkan.exe"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "input.png"
        output_path = temp_dir / "enhanced.png"

        original.save(input_path)

        command = [
            str(exe),
            "-i", str(input_path),
            "-o", str(output_path),
            "-n", "realesrgan-x4plus",
            "-s", "4"
        ]

        result = subprocess.run(
            command,
            cwd=str(REAL_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            details = result.stderr or result.stdout
            raise RuntimeError(details or "The enhancement process failed.")

        if not output_path.exists():
            raise RuntimeError(
                "No output image was created. Check the model files."
            )

        with Image.open(output_path) as image:
            return image.convert("RGB").copy()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">IMAGE PROCESSING</div>
    <h1>Image Enhancement Studio</h1>
    <p>
        Improve photographs and illustrations with image
        enhancement tools. Compare results, inspect resolution,
        and download the processed image.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-label">01 / Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Upload an image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="muted">Supported formats: JPG, PNG and WEBP.</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

original = None

if uploaded is None:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        padding: 2rem;
        border: 1px solid #303744;
        border-radius: 12px;
        background: #171B24;">
        <div style="font-size: 1.05rem; font-weight: 600;">
            Your workspace is ready
        </div>
        <p class="muted">
            Upload a photo or illustration to begin.
            Your processed image will appear here for comparison.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        image_bytes = uploaded.getvalue()
        original = Image.open(BytesIO(image_bytes)).convert("RGB")
        file_hash = hashlib.sha256(image_bytes).hexdigest()

        # Reset results whenever a different image is uploaded
        if st.session_state.get("file_hash") != file_hash:
            st.session_state["file_hash"] = file_hash

            for key in [
                "enhanced_image",
                "cnn_bicubic",
                "cnn_prediction"
            ]:
                st.session_state.pop(key, None)

    except Exception as error:
        st.error(f"Could not open this image: {error}")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    info_col, action_col = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    with info_col:
        st.markdown(
            '<div class="section-label">SOURCE</div>',
            unsafe_allow_html=True
        )

        st.image(
            original,
            caption="Original image",
            use_container_width=True
        )

    with action_col:
        st.markdown(
            '<div class="section-label">IMAGE DETAILS</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Original resolution",
            f"{original.width:,} × {original.height:,}"
        )

        st.metric(
            "File size",
            f"{len(image_bytes) / 1024:.1f} KB"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Enhancement</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="muted">'
            'The general-purpose model enlarges the image by four '
            'times. Results depend on the input image.'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Enhance image",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner("Processing image..."):
                    enhanced = run_realesrgan(original)

                st.session_state["enhanced_image"] = enhanced
                st.success("Processing complete.")

            except subprocess.TimeoutExpired:
                st.error("Processing timed out. Try a smaller image.")

            except Exception as error:
                st.error(f"Enhancement failed: {error}")


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    if "enhanced_image" in st.session_state:
        enhanced = st.session_state["enhanced_image"]

        st.markdown("---")

        st.markdown(
            '<div class="section-label">02 / Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">Before and after</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="muted">'
            'Compare the original and processed images at the same display size.'
            '</div>',
            unsafe_allow_html=True
        )

        before_col, after_col = st.columns(2, gap="large")

        with before_col:
            st.markdown(
                '<div class="image-heading">Original</div>',
                unsafe_allow_html=True
            )
            st.image(original, use_container_width=True)

        with after_col:
            st.markdown(
                '<div class="image-heading">Enhanced</div>',
                unsafe_allow_html=True
            )
            st.image(enhanced, use_container_width=True)

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Output width",
            f"{enhanced.width:,} px"
        )

        metric2.metric(
            "Output height",
            f"{enhanced.height:,} px"
        )

        metric3.metric(
            "Upscaling",
            f"{enhanced.width / original.width:.1f}×"
        )

        buffer = BytesIO()
        enhanced.save(buffer, format="PNG")

        st.download_button(
            "Download enhanced image",
            data=buffer.getvalue(),
            file_name="enhanced_image.png",
            mime="image/png",
            use_container_width=True
        )


    # --------------------------------------------------
    # OPTIONAL CNN EXPERIMENT
    # --------------------------------------------------

    with st.expander("Experimental model comparison"):
        st.markdown(
            "This experiment uses our separately trained convolutional "
            "network. It crops the uploaded image to 32 × 32 pixels, "
            "reduces it to 8 × 8, and compares the reconstruction "
            "against bicubic interpolation. It is intended for learning "
            "and is not a full-resolution photo enhancement method."
        )

        if st.button(
            "Run experimental comparison",
            key="run_cnn"
        ):
            try:
                with st.spinner("Running comparison..."):
                    model, device = load_cnn()

                    crop = ImageOps.fit(
                        original,
                        (32, 32),
                        method=Image.Resampling.LANCZOS
                    )

                    array = np.asarray(
                        crop, dtype=np.float32
                    ) / 255.0

                    tensor = (
                        torch.from_numpy(array)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device)
                    )

                    with torch.no_grad():
                        low = F.interpolate(
                            tensor,
                            size=(8, 8),
                            mode="area"
                        )

                        bicubic = F.interpolate(
                            low,
                            size=(32, 32),
                            mode="bicubic",
                            align_corners=False
                        ).clamp(0, 1)

                        prediction = model(bicubic)

                    st.session_state["cnn_bicubic"] = tensor_to_image(bicubic)
                    st.session_state["cnn_prediction"] = tensor_to_image(prediction)

            except Exception as error:
                st.error(f"Comparison failed: {error}")

        if (
            "cnn_bicubic" in st.session_state
            and "cnn_prediction" in st.session_state
        ):
            ref_col, bic_col, cnn_col = st.columns(3)

            with ref_col:
                st.markdown("**Reference**")
                st.image(
                    original.resize((256, 256), Image.Resampling.NEAREST),
                    use_container_width=True
                )

            with bic_col:
                st.markdown("**Bicubic**")
                st.image(
                    st.session_state["cnn_bicubic"].resize(
                        (256, 256), Image.Resampling.NEAREST
                    ),
                    use_container_width=True
                )

            with cnn_col:
                st.markdown("**Experimental model**")
                st.image(
                    st.session_state["cnn_prediction"].resize(
                        (256, 256), Image.Resampling.NEAREST
                    ),
                    use_container_width=True
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Image Enhancement Studio · Python · PyTorch · Streamlit
</div>
""", unsafe_allow_html=True)
