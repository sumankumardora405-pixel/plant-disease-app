import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. SET UP THE PAGE
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# 2. LOAD YOUR MODEL
# Replace 'model.h5' with the actual path to your model file
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    return model

model = load_my_model()

# 3. DEFINE THE CLASS NAMES
# IMPORTANT: Replace these with your actual class names in the correct order
CLASS_NAMES = [
    "Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Healthy Apple",
    "Corn Common Rust", "Corn Gray Leaf Spot", "Healthy Corn",
    "Potato Early Blight", "Potato Late Blight", "Healthy Potato",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Healthy Tomato"
]

# 4. PREDICTION FUNCTION
def predict_disease(image_data, model):
    # Resize the image to match your model's input size (e.g., 224x224 or 128x128)
    size = (64, 64)  # Change this to your model's expected input size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # Convert image to array
    img_array = np.asarray(image)
    
    # Normalize the image (if your model was trained with normalization)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    
    # Add batch dimension (1, 64, 64, 3)
    img_reshape = normalized_image_array[np.newaxis, ...]
    
    # Make prediction
    prediction = model.predict(img_reshape)
    result_index = np.argmax(prediction) # Get index of highest probability
    return CLASS_NAMES[result_index], prediction[0][result_index]

# 5. USER INTERFACE (UI)
st.title("🌿 Plant Disease Detection")
st.write("Upload an image of a plant leaf, and the AI will identify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("---")
    
    # Prediction Button
    if st.button("Analyze Image"):
        with st.spinner('Analyzing...'):
            label, confidence = predict_disease(image, model)
            
            # Display results
            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence Score:** {confidence*100:.2f}%")

# Footer
st.markdown("---")
st.caption("Note: This is an AI-powered tool. Please consult an expert for critical agricultural decisions.")