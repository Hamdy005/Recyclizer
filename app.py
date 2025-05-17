import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from keras.applications.resnet import preprocess_input

# Constants
IMAGE_SIZE = 224
CLASSES = ["battery", "biological", "cardboard", "clothes", "glass",
           "metal", "paper", "plastic", "shoes", "trash"]

CLASS_TO_CATEGORY = {
    "battery": "Hazardous",
    "biological": "Organic",
    "cardboard": "Recyclable",
    "clothes": "Recyclable",
    "glass": "Recyclable",
    "metal": "Recyclable",
    "paper": "Recyclable",
    "plastic": "Recyclable",
    "shoes": "Recyclable",
    "trash": "Non-recyclable"
}

CATEGORY_INFO = {
    "Hazardous": "⚠️ Handle with care! Take to special recycling centers.",
    "Organic": "🌱 Food and plant waste. Great for making garden soil.",
    "Recyclable": "♻️ Clean it and put it in your recycling bin.",
    "Non-recyclable": "🚫 Goes in regular trash. Try to use less of these items."
}

@st.cache_resource
def load_model():

    model_path = 'resnet50_custom.h5'

    try:

        # First attempt with basic loading

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=None
        )

        return model
    
    except Exception as e:
        st.error(f"First loading attempt failed: {str(e)}")
        
        try:

            # Second attempt with model loading options

            model = tf.keras.models.load_model(
                model_path,
                custom_objects = {},
                options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')

            )

            return model
        
        except Exception as e:

            st.error("Failed to load model with both methods")
            return None


def predict_image(image, model):
    if model is None:
        return None, None, 0.0
    
    # Preprocess image
    img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis = 0)
    img_array = preprocess_input(img_array)
    
    # Predict
    pred = model.predict(img_array, verbose=0)
    
    # Confidence and entropy
    confidence = np.max(pred[0])
    entropy = -np.sum(pred[0] * np.log(pred[0] + 1e-10))
    
    CONFIDENCE_THRESHOLD = 0.5
    ENTROPY_THRESHOLD = 1.5
    
    if confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        return None, None, confidence
    
    class_idx = np.argmax(pred[0])
    class_name = CLASSES[class_idx]
    category = CLASS_TO_CATEGORY[class_name]
    
    # Display "Non-recyclable item" instead of "trash"
    display_name = "Non-recyclable item" if class_name == "trash" else class_name
    
    return display_name, category, confidence


def main():

    st.title("Recyclizer")
    st.markdown("Upload an image of a waste item to classify it.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    model = load_model()
    
    if uploaded_file is not None and model is not None:

        # Load and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Get predictions
        class_name, category, confidence = predict_image(image, model)
        
        # Display image
        st.image(image, channels="BGR", caption="Uploaded Image")
        st.header("🧾 Prediction Results")
        
        if category is None:
            st.warning("⚠️ Unknown waste type!")
            st.write("This item doesn't match any of our known waste categories")
            
        else:

            # Display prediction results
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Category", category)

            with col2:
                # class_name is now display_name from predict_image
                st.metric("Type", class_name)  # Changed label from "Class" to "Type"
            
            st.info(CATEGORY_INFO[category])
            
            # Display confidence with proper formatting
            confidence_pct = float(confidence)  # Ensure it's a float

            if confidence_pct > 0.8:
                st.success(f"Confidence: {confidence_pct:.2%}")

            elif confidence_pct > 0.5:
                st.warning(f"Confidence: {confidence_pct:.2%}")

            else:
                st.error(f"Low confidence: {confidence_pct:.2%}")
            
            # Ensure progress value is between 0 and 1
            st.progress(min(max(confidence_pct, 0.0), 1.0))

if __name__ == "__main__":
    main()
