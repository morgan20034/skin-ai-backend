import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils.symptom_encoder import encode_symptoms
from utils.fusion import fuse_predictions


# ==================================================
# Path Config
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ==================================================
# App Setup
# ==================================================
app = FastAPI(title="Skin AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Starting Skin AI Backend...")


# ==================================================
# Load Models (โหลดครั้งเดียวตอน start server)
# ==================================================
try:
    print("📦 Loading models...")

    image_model = tf.keras.models.load_model(
        MODEL_DIR / "best_model.h5",
        compile=False
    )

    symptom_model = tf.keras.models.load_model(
        MODEL_DIR / "checkbox_model_10class.h5",
        compile=False
    )

    # 🔥 Warmup (กัน request แรกช้า)
    dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    image_model.predict(dummy_img, verbose=0)

    dummy_sym = np.zeros((1, 20))  # ปรับขนาดตาม input symptom model
    symptom_model.predict(dummy_sym, verbose=0)

    print("✅ Models loaded successfully")

except Exception as e:
    print("❌ Model load error:", e)
    raise RuntimeError(f"Model load error: {e}")


# ==================================================
# Load Classes
# ==================================================
try:
    with open(MODEL_DIR / "classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)

    IMAGE_CLASSES = classes["image_model"]["class_names"]
    SYMPTOM_CLASSES = classes["checkbox_model"]["class_names"]
    IMAGE_ID2LABEL = classes["image_model"]["id_to_label"]

except Exception as e:
    print("❌ Classes load error:", e)
    raise RuntimeError(f"Classes load error: {e}")


# ==================================================
# Health Check
# ==================================================
@app.get("/")
def root():
    return {
        "status": "Skin AI Backend is running 🚀",
        "image_classes": len(IMAGE_CLASSES),
        "symptom_classes": len(SYMPTOM_CLASSES)
    }


# ==================================================
# Predict Endpoint
# ==================================================
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    symptoms: str = Form(""),
    areas: str = Form("")
):
    try:
        print("📥 Received prediction request")

        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Empty image file")

        # ---------- Image ----------
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        img_pred = image_model.predict(img_tensor, verbose=0)[0]

        # ---------- Symptoms ----------
        symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]
        area_list = [a.strip() for a in areas.split(",") if a.strip()]

        sym_vec = encode_symptoms(
            selected_symptoms=symptom_list,
            selected_areas=area_list,
            schema_path=str(MODEL_DIR / "symptom_schema.json")
        )

        sym_pred = symptom_model.predict(sym_vec, verbose=0)[0]

        # ---------- Align class size ----------
        sym_full = np.zeros(len(IMAGE_CLASSES), dtype="float32")

        for i, cls in enumerate(SYMPTOM_CLASSES):
            if cls in IMAGE_CLASSES:
                idx = IMAGE_CLASSES.index(cls)
                sym_full[idx] = sym_pred[i]

        # ---------- Fusion ----------
        final_pred = fuse_predictions(
            image_pred=img_pred,
            symptom_pred=sym_full
        )

        top_idx = int(np.argmax(final_pred))
        confidence = float(final_pred[top_idx])

        print("✅ Prediction success:", IMAGE_ID2LABEL[str(top_idx)])

        return {
            "predicted_class": IMAGE_ID2LABEL[str(top_idx)],
            "confidence": confidence,
            "all_scores": {
                IMAGE_ID2LABEL[str(i)]: float(final_pred[i])
                for i in range(len(IMAGE_CLASSES))
            }
        }

    except Exception as e:
        print("❌ BACKEND ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ==================================================
# Local Run
# ==================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
