"""
NutriScan AI - Backend
-----------------------
FastAPI server that loads the trained models (eye x2, nail: Keras;
tongue: PyTorch), runs predictions on uploaded images, and combines
them into a single nutrient-deficiency report (Vitamin A, Vitamin B,
Iron, Zinc).

Run with:
    uvicorn main:app --reload --port 8000

Eye: two photos are taken.
  - whole eye              -> eye_vita model (vitamin A)
  - lower eyelid pulled down -> eye_iron model (iron)
Both expect raw [0,255] RGB input at 224x224.
"""

import io
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from tensorflow.keras.models import load_model

from model_arch import build_organ_model
from tongue_model_arch import HybridCNNViT, tongue_bytes_to_tensor
from config import (
    FULL_MODEL_PATHS,
    WEIGHTS_ONLY_PATHS,
    TONGUE_MODEL_PATH,
    MODEL_NUM_CLASSES,
    IMG_SIZE,
    NAIL_CLASSES,
    TONGUE_CLASSES,
    NAIL_TO_NUTRIENT,
    TONGUE_TO_NUTRIENT,
    TONGUE_SIGNIFICANT_THRESHOLD,
    DEFICIENCY_THRESHOLD,
    EYE_IRON_THRESHOLD,
    EYE_VITA_THRESHOLD,
    IRON_FUSION_WEIGHTS,
)

app = FastAPI(title="NutriScan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Load models once at startup
# ----------------------------------------------------------------------
models = {}          # Keras models: "nail", "eye_iron", "eye_vita"
tongue_model = None  # PyTorch model, loaded separately


@app.on_event("startup")
def load_models():
    global tongue_model

    # Full .keras files -- load directly. safe_mode=False allows custom
    # / Lambda layers that Keras 3 blocks by default.
    for key, path in FULL_MODEL_PATHS.items():
        try:
            model = load_model(path, compile=False, safe_mode=False)
            models[key] = model
            print(f"[OK] Loaded {key} model from {path} -> output shape {model.output_shape}")
        except Exception as e:
            print(f"[WARN] Could not load {key} model from {path}: {e}")
            models[key] = None

    # Weights-only models (none right now, kept for future use)
    for key, weights_path in WEIGHTS_ONLY_PATHS.items():
        try:
            model = build_organ_model(
                num_classes=MODEL_NUM_CLASSES[key],
                model_name=f"{key}_model",
            )
            model.load_weights(weights_path)
            models[key] = model
            print(f"[OK] Loaded {key} model from {weights_path} -> output shape {model.output_shape}")
        except Exception as e:
            print(f"[WARN] Could not load {key} model from {weights_path}: {e}")
            models[key] = None

    # Tongue -- PyTorch state_dict into HybridCNNViT.
    try:
        m = HybridCNNViT(num_classes=len(TONGUE_CLASSES), pretrained=False)
        state_dict = torch.load(TONGUE_MODEL_PATH, map_location="cpu")
        m.load_state_dict(state_dict)
        m.eval()
        tongue_model = m
        print(f"[OK] Loaded tongue model from {TONGUE_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load tongue model from {TONGUE_MODEL_PATH}: {e}")
        tongue_model = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def preprocess_image(file_bytes: bytes, size=IMG_SIZE, normalize: bool = True) -> np.ndarray:
    """Load image bytes -> resized numpy array shaped (1, H, W, 3).

    normalize=True  -> divides by 255.
    normalize=False -> keeps raw [0,255] values (nail + eye models).
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32")
    if normalize:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_single(model_key: str, file_bytes: bytes, class_names: list, normalize: bool = True) -> dict:
    """Run one Keras model on one image."""
    model = models.get(model_key)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"{model_key} model is not loaded. Check backend/models/ and config.py",
        )

    arr = preprocess_image(file_bytes, normalize=normalize)
    preds = model.predict(arr, verbose=0)[0]

    if len(preds) != len(class_names):
        raise HTTPException(
            status_code=500,
            detail=(
                f"{model_key} model outputs {len(preds)} values but "
                f"config.py lists {len(class_names)} class names. "
                "Update the *_CLASSES list in config.py to match."
            ),
        )

    probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    top_class = max(probs, key=probs.get)
    return {"probabilities": probs, "prediction": top_class, "confidence": probs[top_class]}


def predict_tongue(file_bytes: bytes) -> dict:
    """Run the PyTorch Hybrid CNN+ViT tongue model on one image."""
    if tongue_model is None:
        raise HTTPException(
            status_code=503,
            detail="tongue model is not loaded. Check backend/models/ and config.py",
        )

    tensor = tongue_bytes_to_tensor(file_bytes)
    if tensor is None:
        raise HTTPException(status_code=400, detail="Could not read uploaded tongue image.")

    with torch.no_grad():
        logits = tongue_model(tensor)
        preds = torch.softmax(logits, dim=1).cpu().numpy()[0]

    if len(preds) != len(TONGUE_CLASSES):
        raise HTTPException(
            status_code=500,
            detail=(
                f"tongue model outputs {len(preds)} values but "
                f"config.py lists {len(TONGUE_CLASSES)} class names."
            ),
        )

    probs = {TONGUE_CLASSES[i]: float(preds[i]) for i in range(len(TONGUE_CLASSES))}
    top_class = max(probs, key=probs.get)

    possible_signs = sorted(
        [{"class": c, "probability": p} for c, p in probs.items() if p >= TONGUE_SIGNIFICANT_THRESHOLD],
        key=lambda item: -item["probability"],
    )

    return {
        "probabilities": probs,
        "prediction": top_class,
        "confidence": probs[top_class],
        "possible_signs": possible_signs,
    }


def _positive_prob(model_key: str, file_bytes: bytes) -> float:
    """Run one binary eye model, return P(deficient). Handles both
    sigmoid (1 output) and softmax (2 outputs, index 1 = deficient)."""
    model = models.get(model_key)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"{model_key} model is not loaded. Check backend/models/ and config.py",
        )
    arr = preprocess_image(file_bytes, normalize=False)   # raw 0-255, as in eye_meta.json
    out = np.asarray(model.predict(arr, verbose=0)[0]).flatten()
    if out.size == 1:
        return float(out[0])
    if out.size == 2:
        return float(out[1])
    raise HTTPException(
        status_code=500,
        detail=f"{model_key} returned {out.size} outputs, expected 1 or 2.",
    )


def predict_eye_pair(whole_bytes=None, lid_bytes=None) -> dict:
    """whole eye -> vitamin A model, pulled-down lower eyelid -> iron model."""
    if not whole_bytes and not lid_bytes:
        raise HTTPException(status_code=400, detail="Upload at least one eye image.")

    probs, detected, margins = {}, [], {}

    if lid_bytes:
        p = _positive_prob("eye_iron", lid_bytes)
        probs["iron_deficient"] = p
        if p >= EYE_IRON_THRESHOLD:
            detected.append("iron_deficient")
            margins["iron_deficient"] = p / EYE_IRON_THRESHOLD

    if whole_bytes:
        p = _positive_prob("eye_vita", whole_bytes)
        probs["vitamin_a_deficient"] = p
        if p >= EYE_VITA_THRESHOLD:
            detected.append("vitamin_a_deficient")
            margins["vitamin_a_deficient"] = p / EYE_VITA_THRESHOLD

    normal_prob = 1.0 - max(probs.values())
    probs["normal"] = normal_prob

    if detected:
        # If both are positive, the one furthest above its threshold is the main prediction
        prediction = max(margins, key=margins.get)
        confidence = probs[prediction]
    else:
        prediction, confidence = "normal", normal_prob

    return {
        "probabilities": probs,
        "prediction": prediction,
        "confidence": confidence,
        "detected_conditions": detected,
        "healthy_probability": normal_prob,
    }


def fuse_iron(eye=None, nail=None, tongue=None):
    """
    Multimodal weighted fusion for iron (decision-level / late fusion):

        P_Iron = 0.4 * P_Eye + 0.3 * P_Nail + 0.3 * P_Tongue

    Per-modality iron probability:
      eye    -> P(iron_deficient) from the eyelid model
      nail   -> P(Pale) + P(Spoon)   (classes mapped to iron in NAIL_TO_NUTRIENT)
      tongue -> P(pale)              (class mapped to iron in TONGUE_TO_NUTRIENT)

    Only modalities that were actually uploaded take part; their weights are
    re-normalised to sum to 1. Returns None if no modality has iron info.
    """
    parts = {}

    if eye and "iron_deficient" in eye.get("probabilities", {}):
        parts["eye"] = eye["probabilities"]["iron_deficient"]

    if nail:
        parts["nail"] = sum(
            p for label, p in nail["probabilities"].items()
            if NAIL_TO_NUTRIENT.get(label) == "iron"
        )

    if tongue:
        parts["tongue"] = sum(
            p for label, p in tongue["probabilities"].items()
            if TONGUE_TO_NUTRIENT.get(label.lower()) == "iron"
        )

    if not parts:
        return None

    weights = {k: IRON_FUSION_WEIGHTS[k] for k in parts}
    total_w = sum(weights.values())
    weights = {k: w / total_w for k, w in weights.items()}
    fused = sum(weights[k] * parts[k] for k in parts)

    return {
        "score": float(fused),
        "inputs": {k: round(v, 4) for k, v in parts.items()},
        "weights_used": {k: round(w, 4) for k, w in weights.items()},
        "formula": "P_Iron = 0.4*P_Eye + 0.3*P_Nail + 0.3*P_Tongue (weights re-normalised if a modality is missing)",
    }


def combine_results(eye=None, nail=None, tongue=None) -> dict:
    """
    Merge the individual predictions into nutrient-level scores for
    Vitamin A, Vitamin B, Iron and Zinc, then flag likely deficiencies.
    Iron uses weighted multimodal fusion (see fuse_iron); the other
    nutrients keep the max-over-modalities rule.
    """
    scores = {"vitamin_a": 0.0, "vitamin_b": 0.0, "iron": 0.0, "zinc": 0.0}
    contributors = {"vitamin_a": [], "vitamin_b": [], "iron": [], "zinc": []}
    forced = set()   # nutrients flagged by the eye models' own thresholds

    if eye:
        p = eye["probabilities"]
        for cls, nutrient in (("vitamin_a_deficient", "vitamin_a"), ("iron_deficient", "iron")):
            if cls in p:
                scores[nutrient] = max(scores[nutrient], p[cls])
                contributors[nutrient].append("eye")
                if cls in eye["detected_conditions"]:
                    forced.add(nutrient)

    if nail:
        p = nail["probabilities"]
        for label, prob in p.items():
            nutrient = NAIL_TO_NUTRIENT.get(label)
            if nutrient in ("iron", "zinc"):
                scores[nutrient] = max(scores[nutrient], prob)
                contributors[nutrient].append("nail")

    if tongue:
        p = tongue["probabilities"]
        for label, prob in p.items():
            nutrient = TONGUE_TO_NUTRIENT.get(label)
            if nutrient in ("iron", "vitamin_b"):
                scores[nutrient] = max(scores[nutrient], prob)
                contributors[nutrient].append("tongue")

    # Iron: replace the max rule with the weighted multimodal fusion score.
    iron_fusion = fuse_iron(eye, nail, tongue)
    if iron_fusion:
        scores["iron"] = iron_fusion["score"]
        contributors["iron"] = list(iron_fusion["inputs"])

    flags = {
        nutrient: {
            "score": round(score, 4),
            "likely_deficient": score >= DEFICIENCY_THRESHOLD or nutrient in forced,
            "based_on": contributors[nutrient],
        }
        for nutrient, score in scores.items()
    }
    if iron_fusion:
        flags["iron"]["fusion"] = iron_fusion
    return flags


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    all_keys = list(FULL_MODEL_PATHS) + list(WEIGHTS_ONLY_PATHS)
    status = {key: (models.get(key) is not None) for key in all_keys}
    status["tongue"] = tongue_model is not None
    return status


@app.post("/predict/eye")
async def predict_eye_route(
    whole_eye_file: UploadFile = File(None),
    eyelid_file: UploadFile = File(None),
):
    whole = await whole_eye_file.read() if whole_eye_file else None
    lid = await eyelid_file.read() if eyelid_file else None
    return predict_eye_pair(whole, lid)


@app.post("/predict/nail")
async def predict_nail(file: UploadFile = File(...)):
    result = predict_single("nail", await file.read(), NAIL_CLASSES, normalize=False)
    return result


@app.post("/predict/tongue")
async def predict_tongue_route(file: UploadFile = File(...)):
    result = predict_tongue(await file.read())
    return result


@app.post("/predict/all")
async def predict_all(
    eye_whole_file: UploadFile = File(None),
    eye_lid_file: UploadFile = File(None),
    nail_file: UploadFile = File(None),
    tongue_file: UploadFile = File(None),
):
    if not any([eye_whole_file, eye_lid_file, nail_file, tongue_file]):
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    eye_result = None
    if eye_whole_file or eye_lid_file:
        eye_result = predict_eye_pair(
            await eye_whole_file.read() if eye_whole_file else None,
            await eye_lid_file.read() if eye_lid_file else None,
        )

    nail_result = predict_single("nail", await nail_file.read(), NAIL_CLASSES, normalize=False) if nail_file else None
    tongue_result = predict_tongue(await tongue_file.read()) if tongue_file else None

    summary = combine_results(eye_result, nail_result, tongue_result)

    return {
        "individual_predictions": {
            "eye": eye_result,
            "nail": nail_result,
            "tongue": tongue_result,
        },
        "deficiency_summary": summary,
    }


# Serve the frontend directly from FastAPI at http://localhost:8000/
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")