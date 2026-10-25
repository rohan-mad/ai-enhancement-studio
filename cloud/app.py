
import streamlit as st
from PIL import Image
from pathlib import Path
from io import BytesIO
import tempfile
import subprocess
import requests
import zipfile
import hashlib
import shutil
import os

st.set_page_config(
    page_title="Image Enhancement Studio",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #101319;
    color: #E7EAF0;
}
.block-container {
    max-width: 1100px;
    padding-top: 2.5rem;
}
h1, h2, h3 {
    color: #F5F6F8;
    letter-spacing: -0.03em;
}
.muted {
    color: #A7AFBD;
    line-height: 1.6;
}
div.stButton > button {
    min-height: 2.8rem;
    border-radius: 9px;
    font-weight: 600;
}
div.stButton > button[kind="primary"] {
    background: #6366F1;
    border-color: #6366F1;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("Image Enhancement Studio")
st.markdown(
    '<p class="muted">Enhance photographs and illustrations, '
    'compare results, and download processed images.</p>',
    unsafe_allow_html=True
)

# Official Linux release of Real-ESRGAN NCNN-Vulkan
RELEASE_URL = (
    "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/"
    "releases/download/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
)


@st.cache_resource
def prepare_realesrgan():
    """Download and prepare the official Linux executable."""
    root = Path(tempfile.gettempdir()) / "realesrgan-linux"
    extracted = root / "extracted"

    if extracted.exists():
        executables = list(
            extracted.rglob("realesrgan-ncnn-vulkan")
        )
        model_dirs = [
            p for p in extracted.rglob("models")
            if (p / "realesrgan-x4plus.param").exists()
        ]

        if executables and model_dirs:
            os.chmod(executables[0], 0o755)
            return executables[0], model_dirs[0]

    if root.exists():
        shutil.rmtree(root)

    extracted.mkdir(parents=True, exist_ok=True)

    response = requests.get(RELEASE_URL, timeout=180)
    response.raise_for_status()

    archive_path = root / "release.zip"
    archive_path.write_bytes(response.content)

    with zipfile.ZipFile(archive_path) as archive:
        # Reject unexpected paths outside the extraction directory.
        base = extracted.resolve()
        for member in archive.infolist():
            destination = (extracted / member.filename).resolve()
            if not destination.is_relative_to(base):
                raise RuntimeError("Unsafe path in release archive.")
        archive.extractall(extracted)

    executables = list(
        extracted.rglob("realesrgan-ncnn-vulkan")
    )
    model_dirs = [
        p for p in extracted.rglob("models")
        if (p / "realesrgan-x4plus.param").exists()
    ]

    if not executables or not model_dirs:
        raise FileNotFoundError(
            "The Linux executable or model files were not found "
            "in the downloaded release."
        )

    executable = executables[0]
    os.chmod(executable, 0o755)

    return executable, model_dirs[0]


uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image_bytes = uploaded.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    try:
        original = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")
    except Exception:
        st.error("Could not read this image. Try another file.")
        st.stop()

    if st.session_state.get("image_hash") != image_hash:
        st.session_state["image_hash"] = image_hash
        st.session_state.pop("enhanced_bytes", None)

    st.subheader("Original image")
    st.image(original, use_container_width=True)
    st.caption(
        f"Original resolution: {original.width:,} × "
        f"{original.height:,} pixels"
    )

    mode = st.selectbox(
        "Enhancement mode",
        ["Photographs", "Anime and illustrations"]
    )

    model_name = (
        "realesrgan-x4plus"
        if mode == "Photographs"
        else "realesrgan-x4plus-anime"
    )

    st.caption(
        "Processing runs on the hosting server. Large images may "
        "take longer or exceed the free hosting resource limits."
    )

    if st.button(
        "Enhance image",
        type="primary",
        use_container_width=True
    ):
        try:
            with st.spinner(
                "Preparing the enhancement engine..."
            ):
                executable, model_dir = prepare_realesrgan()

            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                input_path = temp / "input.png"
                output_path = temp / "output.png"

                original.save(input_path)

                command = [
                    str(executable),
                    "-i", str(input_path),
                    "-o", str(output_path),
                    "-n", model_name,
                    "-s", "4",
                    "-t", "256",
                    "-m", str(model_dir)
                ]

                with st.spinner("Enhancing your image..."):
                    result = subprocess.run(
                        command,
                        cwd=str(executable.parent),
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                if result.returncode != 0:
                    details = result.stderr or result.stdout
                    raise RuntimeError(
                        details or "The enhancement process failed."
                    )

                if not output_path.exists():
                    raise RuntimeError(
                        "The process finished without creating an image."
                    )

                st.session_state["enhanced_bytes"] = (
                    output_path.read_bytes()
                )

        except subprocess.TimeoutExpired:
            st.error(
                "Processing timed out. Try a smaller image."
            )
        except Exception as error:
            st.error(
                "Enhancement failed. The cloud host may not support "
                "the required Vulkan device or runtime."
            )
            st.code(str(error))

    if "enhanced_bytes" in st.session_state:
        enhanced_bytes = st.session_state["enhanced_bytes"]

        enhanced = Image.open(
            BytesIO(enhanced_bytes)
        ).convert("RGB")

        st.divider()
        st.subheader("Before and after")

        before, after = st.columns(2)

        with before:
            st.image(
                original,
                caption="Original",
                use_container_width=True
            )

        with after:
            st.image(
                enhanced,
                caption="Enhanced",
                use_container_width=True
            )

        st.write(
            f"Output resolution: {enhanced.width:,} × "
            f"{enhanced.height:,} pixels"
        )

        st.download_button(
            "Download enhanced image",
            data=enhanced_bytes,
            file_name="enhanced.png",
            mime="image/png",
            use_container_width=True
        )

else:
    st.markdown(
        '<div class="muted">Upload an image to begin.</div>',
        unsafe_allow_html=True
    )
