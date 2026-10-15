
"""
Config for NutriScan AI backend.
Downloads trained models from Hugging Face.
"""

from huggingface_hub import hf_hub_download

# Hugging Face repository
HF_REPO_ID = "punm12/nutriscan-models"
HF_REPO_TYPE = "model"


def download_model(filename):
    """Download a real trained model and return its local cached path."""
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type=HF_REPO_TYPE,
    )


# ---------------------------------------------------------
# Download the four trained models
# ---------------------------------------------------------

FULL_MODEL_PATHS = {
    "nail": download_model("my_model.keras"),
    "eye_iron": download_model("eye_iron_model.keras"),
    "eye_vita": download_model("eye_vita_model.keras"),
}

WEIGHTS_ONLY_PATHS = {}

MODEL_NUM_CLASSES = {}

# Image size
IMG_SIZE = (224, 224)

# ---------------------------------------------------------
# Eye prediction thresholds
# ---------------------------------------------------------

EYE_IRON_THRESHOLD = 0.44
EYE_VITA_THRESHOLD = 0.42

# ---------------------------------------------------------
# Class labels
# ---------------------------------------------------------

NAIL_CLASSES = [
    "Healthy",
    "Leukonychia",
    "Pale",
    "Spoon",
]

TONGUE_CLASSES = [
    "cracked",
    "healthy",
    "pale",
    "red",
]

# ---------------------------------------------------------
# Download the original PyTorch tongue model
# ---------------------------------------------------------

TONGUE_MODEL_PATH = download_model("best_model (1).pt")

# ---------------------------------------------------------
# Category-to-nutrient mappings
# ---------------------------------------------------------

NAIL_TO_NUTRIENT = {
    "Healthy": "normal",
    "Pale": "iron",
    "Spoon": "iron",
    "Leukonychia": "zinc",
}

TONGUE_TO_NUTRIENT = {
    "healthy": "normal",
    "pale": "iron",
    "red": "vitamin_b",
    "cracked": "vitamin_b",
}

# ---------------------------------------------------------
# Prediction thresholds
# ---------------------------------------------------------

DEFICIENCY_THRESHOLD = 0.5
TONGUE_SIGNIFICANT_THRESHOLD = 0.20

# ---------------------------------------------------------
# Weighted iron fusion
# ---------------------------------------------------------

IRON_FUSION_WEIGHTS = {
    "eye": 0.4,
    "nail": 0.3,
    "tongue": 0.3,
}

