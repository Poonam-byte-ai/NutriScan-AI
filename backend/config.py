"""
Config for NutriScan AI backend.

Nail model: trained via image_dataset_from_directory, which assigns
class indices ALPHABETICALLY by folder name:
    0: Healthy, 1: Leukonychia, 2: Pale, 3: Spoon

Tongue model: trained via torchvision.datasets.ImageFolder, also
alphabetical:
    0: cracked, 1: healthy, 2: pale, 3: red

Eye: TWO separate binary models.
    eye_iron -> lower eyelid (pulled down) photo -> normal / iron_deficient
    eye_vita -> whole eye photo -> normal_full_eye / vitamin_a_deficient
Both take raw RGB float32 0-255 at 224x224 (EfficientNet rescales inside).
"""

# ----------------------------------------------------------------------
# Full .keras files (architecture + weights + config bundled in one file)
# ----------------------------------------------------------------------
FULL_MODEL_PATHS = {
    "nail": "models/my_model.keras",
    "eye_iron": "models/eye_iron_model.keras",
    "eye_vita": "models/eye_vita_model.keras",
}

# No weights-only models anymore (eye moved to full .keras files)
WEIGHTS_ONLY_PATHS = {}

MODEL_NUM_CLASSES = {}

# Image size Keras models expect (width, height)
IMG_SIZE = (224, 224)

# ----------------------------------------------------------------------
# Eye thresholds (from eye_meta.json, "screening" mode)
# ----------------------------------------------------------------------
EYE_IRON_THRESHOLD = 0.44
EYE_VITA_THRESHOLD = 0.42

# ----------------------------------------------------------------------
# Class label order
# ----------------------------------------------------------------------
NAIL_CLASSES = ["Healthy", "Leukonychia", "Pale", "Spoon"]

TONGUE_CLASSES = ["cracked", "healthy", "pale", "red"]

# ----------------------------------------------------------------------
# Tongue model — separate PyTorch pipeline (Hybrid CNN+ViT)
# ----------------------------------------------------------------------
TONGUE_MODEL_PATH = "models/best_model (1).pt"

# ----------------------------------------------------------------------
# Category -> underlying nutrient mapping
# ----------------------------------------------------------------------
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

# Probability threshold above which a nutrient is flagged as "likely deficient"
DEFICIENCY_THRESHOLD = 0.5

# Any tongue class at or above this probability is reported as a possible sign
TONGUE_SIGNIFICANT_THRESHOLD = 0.20

# ----------------------------------------------------------------------
# Multimodal weighted fusion for IRON (decision-level / late fusion)
#   P_Iron = 0.4 * P_Eye + 0.3 * P_Nail + 0.3 * P_Tongue
# If a modality is not uploaded, the remaining weights are re-normalised
# so they still sum to 1.
# ----------------------------------------------------------------------
IRON_FUSION_WEIGHTS = {"eye": 0.4, "nail": 0.3, "tongue": 0.3}