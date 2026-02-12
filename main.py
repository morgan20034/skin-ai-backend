import os
import json
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils.image_preprocess import preprocess_image
from utils.symptom_encoder import encode_symptoms
from utils.fusion import fuse_predictions


# ==================================================
# App Setup
# ==================================================
app = FastAPI(title="Skin AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Starting Skin AI Backend...")


# ==================================================
# Load Models
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")
SYMPTOM_MODEL_PATH = os.path.join(BASE_DIR, "models", "checkbox_model_10class.h5")
SCHEMA_PATH = os.path.join(BASE_DIR, "models", "symptom_schema.json")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.json")

try:
    print("📦 Loading models...")

    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    symptom_model = tf.keras.models.load_model(SYMPTOM_MODEL_PATH)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_config = json.load(f)

    image_class_names = classes_config["image_model"]["class_names"]

    print("✅ Models loaded successfully")

except Exception as e:
    print("❌ Model loading failed:", e)
    raise e


# ==================================================
# Health Check (Render ใช้เช็คว่า service ทำงานไหม)
# ==================================================
@app.get("/")
def health_check():
    return {"status": "Skin AI backend is running 🚀"}


# ==================================================
# Predict API
# ==================================================
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    symptoms: str = Form(""),
    areas: str = Form("")
):
    try:
        print("📥 Received request")

        # 1️⃣ Read image
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Empty image file")

        # 2️⃣ Image model (11 classes)
        image_tensor = preprocess_image(image_bytes)
        image_probs = image_model.predict(image_tensor)
        image_probs = np.squeeze(image_probs)

        if image_probs.shape[0] != 11:
            raise ValueError(f"Image model output shape error: {image_probs.shape}")

        print("🖼 Image prediction OK")

        # 3️⃣ Parse symptoms & areas
        try:
            selected_symptoms = json.loads(symptoms) if symptoms else []
        except:
            selected_symptoms = [s for s in symptoms.split(",") if s]

        try:
            selected_areas = json.loads(areas) if areas else []
        except:
            selected_areas = [a for a in areas.split(",") if a]

        print("🧠 Symptoms:", selected_symptoms)
        print("📍 Areas:", selected_areas)

        # 4️⃣ Symptom model (10 classes)
        symptom_tensor = encode_symptoms(
            selected_symptoms,
            selected_areas,
            SCHEMA_PATH
        )

        symptom_probs = symptom_model.predict(symptom_tensor)
        symptom_probs = np.squeeze(symptom_probs)

        if symptom_probs.shape[0] != 10:
            raise ValueError(f"Symptom model output shape error: {symptom_probs.shape}")

        print("🧾 Symptom prediction OK")

        # 5️⃣ Fusion
        fused_probs = fuse_predictions(
            image_pred=image_probs,
            symptom_pred=symptom_probs
        )

        print("🔀 Fusion OK")

        # 6️⃣ Sort result
        class_indices = np.argsort(fused_probs)[::-1]
        top_index = int(class_indices[0])
        top_score = float(fused_probs[top_index])

        if top_score < 0.4:
            recommendation = "โมเดลไม่มั่นใจ แนะนำพบแพทย์ผิวหนัง"
        else:
            recommendation = "ผลเป็นการประเมินเบื้องต้น ควรพบแพทย์เพื่อยืนยัน"

        print("✅ Prediction success:", image_class_names[top_index])

        return {
            "top_prediction": {
                "class_index": top_index,
                "class_name": image_class_names[top_index],
                "score": top_score
            },
            "all_scores": {
                image_class_names[i]: float(fused_probs[i])
                for i in range(len(fused_probs))
            },
            "recommendation": recommendation
        }

    except Exception as e:
        print("❌ BACKEND ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ==================================================
# Local Run (สำหรับรันในเครื่องเท่านั้น)
# ==================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
