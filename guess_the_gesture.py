import cv2
import numpy as np
import tensorflow as tf

# ====== Load Model and Classes ======
MODEL_PATH = "augmented_cnn_model.keras"   
CLASS_PATH = "class_names.npy"

print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
class_names = np.load(CLASS_PATH)
print(f" Model loaded successfully. Classes: {class_names}")

# ====== Webcam Setup ======
cap = cv2.VideoCapture(0)
img_height, img_width = 180, 180

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural mirror view
    frame = cv2.flip(frame, 1)

    # Draw a guide box 
    h, w, _ = frame.shape
    box_size = 250
    x1, y1 = (w - box_size)//2, (h - box_size)//2
    x2, y2 = x1 + box_size, y1 + box_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (img_width, img_height))
    roi_norm = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_norm, axis=0)

    # Prediction
    preds = model.predict(roi_input, verbose=0)
    class_id = np.argmax(preds)
    confidence = np.max(preds)
    label = f"{class_names[class_id]} ({confidence*100:.1f}%)"

    # Display label
    cv2.putText(frame, label, (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow(" Hand Sign Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

