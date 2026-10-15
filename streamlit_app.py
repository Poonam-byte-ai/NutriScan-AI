
import os
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

st.set_page_config(
    page_title="NutriScan AI",
    page_icon="🔬",
    layout="wide",
)

st.title("NutriScan AI")
st.write(
    "Upload eye, nail, and/or tongue images to run predictions "
    "using your trained models."
)
st.caption(
    "Research prototype only. Results are not a medical diagnosis."
)


@st.cache_resource(show_spinner="Loading NutriScan trained models...")
def load_backend():
    if not BACKEND_DIR.is_dir():
        raise FileNotFoundError(
            "The backend folder was not found in the repository root."
        )

    # Your existing config uses paths relative to backend/.
    os.chdir(BACKEND_DIR)

    backend_path = str(BACKEND_DIR)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    import main as backend

    # Streamlit doesn't automatically run FastAPI startup events.
    # Explicitly load the actual trained models.
    backend.load_models()

    return backend


with st.spinner("Preparing the trained models..."):
    try:
        backend = load_backend()
        st.success("Model-loading process completed.")
    except Exception as exc:
        st.error(f"Could not initialize NutriScan: {exc}")
        st.stop()


with st.expander("Model loading status", expanded=False):
    statuses = {
        name: model is not None
        for name, model in backend.models.items()
    }
    statuses["tongue"] = backend.tongue_model is not None

    for name, loaded in statuses.items():
        st.write(f"{'✅' if loaded else '❌'} {name}: "
                 f"{'loaded' if loaded else 'not loaded'}")


st.subheader("Upload images")

col1, col2 = st.columns(2)

with col1:
    whole_eye = st.file_uploader(
        "Whole-eye image (vitamin A)",
        type=["jpg", "jpeg", "png"],
        key="whole_eye",
    )
    eyelid = st.file_uploader(
        "Lower-eyelid image (iron)",
        type=["jpg", "jpeg", "png"],
        key="eyelid",
    )

with col2:
    nail = st.file_uploader(
        "Nail image",
        type=["jpg", "jpeg", "png"],
        key="nail",
    )
    tongue = st.file_uploader(
        "Tongue image",
        type=["jpg", "jpeg", "png"],
        key="tongue",
    )


if st.button("Run NutriScan prediction", type="primary"):
    if not any([whole_eye, eyelid, nail, tongue]):
        st.warning("Please upload at least one image.")
        st.stop()

    try:
        eye_result = None
        nail_result = None
        tongue_result = None

        if whole_eye or eyelid:
            eye_result = backend.predict_eye_pair(
                whole_eye.getvalue() if whole_eye else None,
                eyelid.getvalue() if eyelid else None,
            )

        if nail:
            nail_result = backend.predict_single(
                "nail",
                nail.getvalue(),
                backend.NAIL_CLASSES,
                normalize=False,
            )

        if tongue:
            tongue_result = backend.predict_tongue(
                tongue.getvalue()
            )

        summary = backend.combine_results(
            eye_result,
            nail_result,
            tongue_result,
        )

        st.subheader("Deficiency summary")
        for nutrient, details in summary.items():
            score = details.get("score", 0.0)
            likely = details.get("likely_deficient", False)

            left, right = st.columns([3, 2])
            left.write(nutrient.replace("_", " ").title())
            right.write(
                f"{'Flagged for further assessment' if likely else 'Not flagged'}"
                f" — model score: {score:.1%}"
            )
            st.progress(min(max(float(score), 0.0), 1.0))

        with st.expander("Individual model predictions"):
            st.json({
                "eye": eye_result,
                "nail": nail_result,
                "tongue": tongue_result,
            })

        with st.expander("Full fusion output"):
            st.json(summary)

    except Exception as exc:
        st.error(
            "Prediction failed. Check the app logs and confirm that "
            f"all required model files loaded correctly. Details: {exc}"
        )
