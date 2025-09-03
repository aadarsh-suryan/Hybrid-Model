import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# ✅ Load your trained model (update path if needed)
model = tf.keras.models.load_model(r"C:\Users\aadar\Desktop\Hybrid Model\best_model.h5")

# ✅ Class names used during training
class_names = ['Calculus', 'Dental Carirs', 'Ulcer']

# ✅ Page config
st.set_page_config(page_title="Mouth Ulcer Detection", layout="centered")
st.title("🦷 Mouth Ulcer & Dental Calculus Detection using Deep Learning")
st.markdown("Upload a clear image of the mouth/gums to detect *Healthy, **Ulcer, or **Calculus* conditions.")

# ✅ Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    # ✅ Resize, normalize, convert
    img = cv2.resize(img_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = img_rgb / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # ✅ Show original image
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # ✅ Prediction
    prediction = model.predict(img_input)[0]
    class_index = np.argmax(prediction)
    label = class_names[class_index]
    confidence = prediction[class_index] * 100

    # ✅ Show result
    st.markdown(
        f"<h3 style='color: pink;'>🧠 Prediction: {label} ({confidence:.2f}%)</h3>",
        unsafe_allow_html=True
    )

    # ✅ Report text
    report_text = f"""
🦷 *Mouth Ulcer Detection Report*

*Prediction:* {label}  
*Confidence:* {confidence:.2f}%

Note: This result is AI-generated. Please consult a dentist for a medical opinion.
"""

    # ✅ Download button
    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="ulcer_detection_report.txt",
        mime="text/plain"
    )