"""Streamlit web interface for the HCC MobileNet-V2 research prototype.

Supports:
- NIfTI CT volumes (.nii/.nii.gz) with slice selection
- PNG/JPG/BMP/TIFF single-slice images
- Optional .pth model upload when model weights are not committed to the repo

Research/educational use only; not a clinical diagnostic system.
"""

import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import nibabel as nib
from PIL import Image

from inference import LiverCancerClassifier


st.set_page_config(
    page_title="HCC AI | CT Analysis",
    page_icon="HCC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    .hero { padding: 1.5rem 0 1rem 0; }
    .hero h1 { font-size: 2.6rem; margin-bottom: .35rem; }
    .hero p { color: #6b7280; font-size: 1.05rem; }
    .result-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; background: #fafafa; }
    .metric-label { color: #6b7280; font-size: .85rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .disclaimer { color: #6b7280; font-size: .78rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>HCC AI Analysis</h1>
      <p>MobileNet-V2 based CT-slice classification and research visualization.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def get_nifti_volume(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.lower().endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        path = tmp.name
    try:
        nii = nib.load(path)
        volume = np.rot90(np.asarray(nii.get_fdata()), 1, axes=(0, 1))
        return volume.astype(np.float32)
    finally:
        Path(path).unlink(missing_ok=True)


def get_image_slice(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    pixels = np.asarray(image, dtype=np.float32)

    # PNG/JPG files do not contain CT Hounsfield Units. For the research demo,
    # map 8-bit grayscale into the same nominal range used by the training
    # preprocessing before HU windowing + CLAHE.
    if pixels.max() > pixels.min():
        pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min())
    pixels = pixels * 225.0 - 75.0
    return pixels.astype(np.float32)


def prepare_model_dir(uploaded_model):
    """Return a model directory containing model_info.json and optional weights."""
    repo_model_dir = Path("models")
    has_repo_weights = any(
        (repo_model_dir / name).exists()
        for name in ("liver_cancer_model_full.pth", "mobilenet_best.pth", "liver_cancer_model.pth")
    )

    if not uploaded_model:
        if has_repo_weights:
            return repo_model_dir, None
        return None, None

    temp_dir = Path(tempfile.mkdtemp(prefix="hcc_model_"))
    info_src = repo_model_dir / "model_info.json"
    if not info_src.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise FileNotFoundError("models/model_info.json is missing from the repository.")
    shutil.copy2(info_src, temp_dir / "model_info.json")

    model_name = uploaded_model.name
    if not model_name.lower().endswith((".pth", ".pt", ".ckpt")):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("Please upload a PyTorch model file (.pth, .pt, or .ckpt).")

    with open(temp_dir / model_name, "wb") as f:
        f.write(uploaded_model.getbuffer())

    # inference.py checks these conventional filenames.
    target = temp_dir / "liver_cancer_model_full.pth"
    if not target.exists():
        shutil.copy2(temp_dir / model_name, target)

    return temp_dir, temp_dir


@st.cache_resource(show_spinner="Loading MobileNet-V2 model...")
def load_classifier(model_dir_str):
    return LiverCancerClassifier(model_dir=model_dir_str)


with st.sidebar:
    st.header("CT Scan")
    input_type = st.radio("Input type", ["NIfTI volume", "PNG/JPG image"])

    if input_type == "NIfTI volume":
        uploaded = st.file_uploader(
            "Upload CT volume",
            type=["nii", "gz"],
            help="Supported formats: .nii and .nii.gz",
        )
    else:
        uploaded = st.file_uploader(
            "Upload CT slice image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            help="PNG/JPG is supported for single-slice research testing.",
        )

    st.divider()
    st.header("Model")
    model_upload = st.file_uploader(
        "Upload model weights if required",
        type=["pth", "pt", "ckpt"],
        help="The repository currently contains model metadata but not the binary .pth weights. Upload the trained weights here unless you commit them separately.",
    )
    st.caption("MobileNet-V2")
    st.caption("Classes: Background · Liver · Tumor")

if not uploaded:
    st.info("Choose NIfTI or PNG/JPG from the sidebar and upload an input image.")
    st.markdown("### What this demo does")
    st.markdown(
        """
        1. Loads a CT volume or single CT slice image.
        2. Applies the project's preprocessing pipeline.
        3. Runs the trained MobileNet-V2 classifier.
        4. Displays prediction, confidence, class probabilities, and visualization.
        """
    )
    st.warning("Research/educational prototype only. This application is not intended for clinical diagnosis.")
    st.stop()

try:
    if input_type == "NIfTI volume":
        volume = get_nifti_volume(uploaded)
        if volume.ndim != 3:
            st.error(f"Expected a 3D CT volume, received shape {volume.shape}.")
            st.stop()
        st.success(f"Loaded {uploaded.name} — volume shape: {volume.shape}")
        max_slice = volume.shape[2] - 1
        slice_idx = st.slider("CT slice", 0, max_slice, max_slice // 2)
        ct_slice = volume[:, :, slice_idx]
        display_label = f"Slice {slice_idx} / {max_slice}"
    else:
        ct_slice = get_image_slice(uploaded)
        slice_idx = 0
        display_label = "Single image"
        st.success(f"Loaded {uploaded.name} — image shape: {ct_slice.shape}")
        st.image(ct_slice, caption="Uploaded grayscale slice", clamp=True, use_container_width=True)
except Exception as exc:
    st.error(f"Could not read the uploaded input: {exc}")
    st.stop()

try:
    model_dir, temp_model_dir = prepare_model_dir(model_upload)
    if model_dir is None:
        st.error("Model inference cannot start because the trained .pth weights are not present.")
        st.info("Upload your trained liver_cancer_model_full.pth (or mobilenet_best.pth) in the Model section of the sidebar, or commit the binary model artifact to the deployment environment.")
        st.stop()

    classifier = load_classifier(str(model_dir))
    result = classifier.predict_slice(ct_slice)
except Exception as exc:
    st.error(f"Model inference failed: {exc}")
    st.info("Verify that the uploaded weights match the MobileNet-V2 architecture and the model_info.json configuration.")
    st.stop()
finally:
    if temp_model_dir is not None:
        shutil.rmtree(temp_model_dir, ignore_errors=True)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="result-card"><div class="metric-label">Prediction</div><div class="metric-value">' + str(result["predicted_class"]) + '</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="result-card"><div class="metric-label">Confidence</div><div class="metric-value">' + f'{result["confidence"]:.2f}%' + '</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="result-card"><div class="metric-label">Input</div><div class="metric-value">' + display_label + '</div></div>', unsafe_allow_html=True)

st.subheader("Class probabilities")
prob_cols = st.columns(len(result["probabilities"]))
for column, (name, probability) in zip(prob_cols, result["probabilities"].items()):
    with column:
        st.metric(name, f"{probability * 100:.2f}%")
        st.progress(float(probability))

st.subheader("CT visualization")
preprocessed = classifier.preprocess_ct_slice(ct_slice)
pred_class = result["predicted_class_id"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
axes[0].imshow(preprocessed, cmap="bone")
axes[0].set_title("Preprocessed CT")
axes[0].axis("off")

axes[1].imshow(preprocessed, cmap="bone")
if pred_class > 0:
    overlay_colors = {1: [0, 0, 1], 2: [1, 0, 0]}
    overlay = np.zeros((*preprocessed.shape, 4))
    overlay[:, :, :3] = overlay_colors[pred_class]
    overlay[:, :, 3] = 0.30
    axes[1].imshow(overlay)
axes[1].set_title(f"Prediction: {result['predicted_class']}")
axes[1].axis("off")

classes = list(result["probabilities"].keys())
probs = [result["probabilities"][c] * 100 for c in classes]
axes[2].barh(classes, probs)
axes[2].set_xlim(0, 100)
axes[2].set_xlabel("Probability (%)")
axes[2].set_title("Class probabilities")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown(
    '<div class="disclaimer">Research and educational prototype only. Results must not be used for medical decision-making and require appropriate clinical validation and oversight.</div>',
    unsafe_allow_html=True,
)
