import numpy as np


def fuse_predictions(
    image_pred: np.ndarray,
    symptom_pred: np.ndarray
) -> np.ndarray:

    IMAGE_WEIGHT = 0.6
    SYMPTOM_WEIGHT = 0.4

    image_pred = np.asarray(image_pred, dtype="float32").flatten()
    symptom_pred = np.asarray(symptom_pred, dtype="float32").flatten()

    if image_pred.shape[0] != 11:
        raise ValueError(
            f"Image model must have 11 classes, got {image_pred.shape}"
        )

    # align symptom (เพิ่ม normal_skin index 6)
    if symptom_pred.shape[0] == 10:
        symptom_pred = np.insert(symptom_pred, 6, 0.0)

    if image_pred.shape != symptom_pred.shape:
        raise ValueError(
            f"Class mismatch after alignment: image={image_pred.shape}, symptom={symptom_pred.shape}"
        )

    # normalize
    image_pred = image_pred / (np.sum(image_pred) + 1e-8)
    symptom_pred = symptom_pred / (np.sum(symptom_pred) + 1e-8)

    # ----------------------------------------
    # 🔥 SMART ADAPTIVE FUSION
    # ----------------------------------------

    image_top = np.argmax(image_pred)
    symptom_top = np.argmax(symptom_pred)

    image_conf = image_pred[image_top]
    symptom_conf = symptom_pred[symptom_top]

    # ถ้า model ไม่เห็นตรงกัน
    if image_top != symptom_top:

        # ถ้า symptom มั่นใจมากกว่า 0.45
        if symptom_conf > 0.45:
            IMAGE_WEIGHT = 0.4
            SYMPTOM_WEIGHT = 0.6

        # ถ้า image มั่นใจสูงเกิน 0.7 ให้ image นำ
        elif image_conf > 0.7:
            IMAGE_WEIGHT = 0.75
            SYMPTOM_WEIGHT = 0.25

    fused = (
        IMAGE_WEIGHT * image_pred +
        SYMPTOM_WEIGHT * symptom_pred
    )

    fused = fused / (np.sum(fused) + 1e-8)

    return fused