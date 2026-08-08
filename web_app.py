"""Web interface for the HCC MobileNet-V2 research prototype.

Run locally with:
    streamlit run web_app.py

The app reuses the existing LiverCancerClassifier from inference.py and
supports NIfTI (.nii/.nii.gz) CT volumes. It is a research/educational demo,
not a clinical diagnostic system.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import nibabel as nib

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

@st.cache_resource(show_spinner="Loading MobileNet-V2 model...")
def load_classifier():
    return LiverCancerClassifier(model_dir="./models")


def get_volume(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.lower().endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        path = tmp.name
    try:
        nii = nib.load(path)
        volume = np.rot90(np.asarray(nii.get_fdata()))
        return volume
    finally:
        Path(path).unlink(missing_ok=True)


with st.sidebar:
    st.header("CT Scan")
    uploaded = st.file_uploader(
        "Upload a NIfTI CT volume",
        type=["nii", "gz"],
        help="Supported formats: .nii and .nii.gz",
    )
    st.divider()
    st.caption("Model")
    st.write("MobileNet-V2")
    st.caption("Classes")
    st.write("Background · Liver · Tumor")

if not uploaded:
    st.info("Upload a .nii or .nii.gz CT volume from the sidebar to begin analysis.")
    st.markdown("### What this demo does")
    st.markdown(
        """
        1. Loads a CT volume in NIfTI format.
        2. Selects an individual axial slice.
        3. Applies the project's HU-windowing and CLAHE preprocessing.
        4. Runs the trained MobileNet-V2 classifier.
        5. Displays the prediction, confidence, probabilities, and visualization.
        """
    )
    st.warning("Research/educational prototype only. This application is not intended for clinical diagnosis.")
    st.stop()

try:
    volume = get_volume(uploaded)
except Exception as exc:
    st.error(f"Could not read the CT volume: {exc}")
    st.stop()

if volume.ndim != 3:
    st.error(f"Expected a 3D CT volume, received shape {volume.shape}.")
    st.stop()

st.success(f"Loaded {uploaded.name} — volume shape: {volume.shape}")

max_slice = volume.shape[2] - 1
slice_idx = st.slider("CT slice", 0, max_slice, max_slice // 2)
ct_slice = volume[:, :, slice_idx]

try:
    classifier = load_classifier()
    result = classifier.predict_slice(ct_slice)
except Exception as exc:
    st.error(f"Model inference failed: {exc}")
    st.info("Make sure the model artifacts are present in the repository's models/ directory.")
    st.stop()

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="result-card"><div class="metric-label">Prediction</div><div class="metric-value">' + str(result["predicted_class"]) + '</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="result-card"><div class="metric-label">Confidence</div><div class="metric-value">' + f'{result["confidence"]:.2f}%' + '</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="result-card"><div class="metric-label">Slice</div><div class="metric-value">' + f'{slice_idx} / {max_slice}' + '</div></div>', unsafe_allow_html=True)

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
