import streamlit as st
import tensorflow as tf 
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Set page title
st.title("Brain Tumor Classification")
st.write("Upload an MRI image and let the model predict the tumor type.")

# Load your trained model
model = load_model('pretrained_model.h5')  # replace with your .h5 filename

# Define class names (replace with your actual class labels)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI.", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0

    if img_array.shape[-1] == 4:  # Remove alpha channel if present
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence_score = float(np.max(prediction)) * 100

    st.markdown(f"### 🧠 Predicted Tumor Type: `{predicted_class}`")
    st.markdown(f"**Confidence Score:** {confidence_score:.2f}%")


    st.write("#### All Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2%}")

