# NutriScan AI

Predicts likely Vitamin A, Vitamin B, Iron, and Calcium deficiencies from
photos of the eye, nail, and tongue, using three separately trained Keras
models.

## 1. Add your models

Copy your trained `.h5`/`.keras` files into `backend/models/`:

```
backend/models/eye_model.h5
backend/models/nail_model.h5
backend/models/tongue_model.h5
```

(Different filenames or `.keras` format? Just update `MODEL_PATHS` in
`backend/config.py`.)

## 2. Fix the class label order (important!)

Open `backend/config.py` and read the big comment at the top. You must set
`EYE_CLASSES`, `NAIL_CLASSES`, and `TONGUE_CLASSES` to match the exact
index order Keras used during training — this is the alphabetical order of
your training folder names, or whatever `train_generator.class_indices`
printed for you. If this is wrong, the app runs fine but mislabels results.

Also check `IMG_SIZE` matches the input shape your models were trained on
(e.g. 224x224, 150x150, etc).

## 3. Install dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Run the backend (also serves the frontend)

```bash
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000/** in your browser — the FastAPI server
serves the frontend directly, so you don't need a separate web server.

## 5. Use it

Upload an eye photo, a nail photo, and/or a tongue photo (you can submit
just one or two if you don't have all three), click **Analyze**, and you'll
get:

- A per-nutrient summary (Vitamin A, Vitamin B, Iron, Calcium) with a
  confidence score and a flag if it crosses the deficiency threshold
  (default 50%, adjustable via `DEFICIENCY_THRESHOLD` in `config.py`).
- The raw per-model prediction breakdown (all class probabilities).

## API reference

| Endpoint | Method | Body | Purpose |
|---|---|---|---|
| `/predict/eye` | POST | `file` (image) | Eye model only |
| `/predict/nail` | POST | `file` (image) | Nail model only |
| `/predict/tongue` | POST | `file` (image) | Tongue model only |
| `/predict/all` | POST | `eye_file`, `nail_file`, `tongue_file` (any subset) | Combined report |
| `/health` | GET | — | Check which models loaded successfully |

## Notes / things worth double-checking

- The `Leukonychia → Calcium` mapping is a simplification for this project;
  clinically it's also linked to zinc deficiency or minor trauma. Worth a
  caveat in your report/demo if this is for an academic submission.
- This is a screening prototype, not a diagnostic tool — the frontend
  footer says so, keep that disclaimer if you present/deploy this.
- CORS is wide open (`allow_origins=["*"]`) for easy local development.
  Lock this down before deploying anywhere public.
