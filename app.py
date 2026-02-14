# app.py
# Mushroom Species Classification Web Application
# Streamlit-based deployment for deep learning image classifier

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Multiply
import os

# Configure page
st.set_page_config(
    page_title="Mushroom Species Classifier",
    page_icon="🍄",
    layout="wide"
)

# Class labels
CLASS_NAMES = [
    "Agaricus",
    "Amanita",
    "Boletus",
    "Cortinarius",
    "Entoloma",
    "Hygrocybe",
    "Lactarius",
    "Russula",
    "Suillus"
]

IMG_SIZE = (224, 224)

def build_model(num_classes, trainable_base=False):
    """Build EfficientNetB0-based model with Squeeze-Excitation attention."""

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    base_model.trainable = trainable_base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Squeeze-Excitation Block
    se = Dense(128, activation='relu')(x)
    se = Dense(1280, activation='sigmoid')(se)
    x = Multiply()([x, se])   # Channel attention

    # Classifier Head
    x = Dense(512, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(256, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Model loading with caching to avoid reloading on every interaction
@st.cache_resource
def load_classification_model():
    """Load the pre-trained Keras model"""
    try:
        # Rebuild the model architecture
        model = build_model(num_classes=9, trainable_base=False)

        # Load the weights (replace with your actual weights path)
        model.load_weights('best_model_stage1_weights.h5')

        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def preprocess_image(img):
    """
    Preprocess uploaded image for model inference
    - Resize to 224x224
    - Convert to array
    - Apply EfficientNet preprocessing (CRITICAL: must match training)
    """
    try:
        # Resize image
        img = img.resize((224, 224))
        
        # Convert to array
        img_array = image.img_to_array(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Use EfficientNet-specific preprocessing (imported at top)
        # This applies [-1, 1] normalization
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_mushroom(model, img_array):
    """
    Make prediction using the loaded model
    Returns class probabilities
    """
    try:
        predictions = model.predict(img_array, verbose=0)
        return predictions[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def predict_with_tta(model, img, n_aug=10):
    """
    Lightweight TTA for Streamlit deployment
    - Resize FIRST (to 224x224), THEN augment
    - Conservative augmentation matching training TTA strategy
    - Horizontal flip + small brightness variations
    """
    preds = []
    
    # Convert PIL to numpy for easier manipulation
    img_resized = img.resize((224, 224))
    
    for i in range(n_aug):
        aug_img = img_resized.copy()
        
        # 50% chance horizontal flip (mushrooms are symmetric)
        if i % 2 == 0:
            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Small brightness adjustment (±3% like training TTA)
        if i > 0:  # Keep first prediction clean
            brightness_factor = np.random.uniform(0.97, 1.03)
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(brightness_factor)
        
        # Preprocess and predict
        x = preprocess_image(aug_img)
        if x is not None:
            pred = model.predict(x, verbose=0)[0]
            preds.append(pred)
    
    # Return averaged predictions
    return np.mean(preds, axis=0)

# Main application
def main():
    # Header section
    st.title("🍄 Mushroom Species Classification")
    st.markdown("""
    ### Deep Learning-Based Mushroom Identifier
    Upload an image of a mushroom to identify its species. This model can classify 9 different mushroom genera.
    
    **Model:** EfficientNetB0 + Squeeze-Excitation | **Test Accuracy:** 87.77% | **Training:** Transfer Learning + TTA
    """)
    
    st.markdown("---")
    
    # Load model
    model = load_classification_model()
    
    # Sidebar with model info
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown("""
        **Architecture:**
        - Base: EfficientNetB0 (ImageNet pre-trained)
        - Attention: Squeeze-Excitation blocks
        - Classifier: 2-layer with BatchNorm
        
        **Performance:**
        - Test Accuracy: 87.77%
        - Validation Accuracy: 87.33%
        - Improvement over baseline: +20.71%
        
        **Dataset:**
        - 9 mushroom genera
        - 6,711 total images
        - Training: 4,695 images
        """)
        
        st.markdown("---")
        st.warning("⚠️ **Safety Notice**: Never eat wild mushrooms without expert mycologist identification. Many species are deadly poisonous.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a mushroom image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a mushroom for classification"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            img = Image.open(uploaded_file)
            
            # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Uploaded Image")
                st.image(img, use_container_width=True)
                st.caption(f"Image size: {img.size[0]} × {img.size[1]} pixels")
            
            with col2:
                st.subheader("🔍 Classification Results")
                
                # Make prediction with TTA
                with st.spinner("Analyzing mushroom with Test-Time Augmentation..."):
                    predictions = predict_with_tta(model, img, n_aug=10)
                
                if predictions is not None:
                    # Get predicted class
                    predicted_class_idx = np.argmax(predictions)
                    predicted_class_name = CLASS_NAMES[predicted_class_idx]
                    confidence = predictions[predicted_class_idx] * 100
                    
                    # Display prediction prominently
                    st.success(f"**Predicted Species:** {predicted_class_name}")
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.2f}%"
                    )
                    
                    # Confidence interpretation
                    if confidence >= 90:
                        st.info("🟢 **High confidence** - Model is very certain")
                    elif confidence >= 70:
                        st.warning("🟡 **Moderate confidence** - Model has some uncertainty")
                    else:
                        st.error("🔴 **Low confidence** - Image may be unclear or ambiguous")
                    
                    # Display confidence scores for all classes
                    st.markdown("---")
                    st.subheader("📊 Confidence Scores (All Classes)")
                    
                    # Create sorted list of predictions
                    pred_list = [(CLASS_NAMES[i], predictions[i] * 100) 
                                for i in range(len(CLASS_NAMES))]
                    pred_list.sort(key=lambda x: x[1], reverse=True)
                    
                    # Display progress bars for top 5 classes only (cleaner UI)
                    st.markdown("**Top 5 Predictions:**")
                    for class_name, prob in pred_list[:5]:
                        # Highlight the predicted class
                        if class_name == predicted_class_name:
                            st.markdown(f"**🏆 {class_name}**")
                        else:
                            st.markdown(f"{class_name}")
                        st.progress(prob / 100)
                        st.caption(f"{prob:.2f}%")
                        st.markdown("")
                    
                    # Expandable section for all classes
                    with st.expander("Show all 9 class probabilities"):
                        for class_name, prob in pred_list:
                            st.text(f"{class_name}: {prob:.2f}%")
            
            # Additional information section
            st.markdown("---")
            st.info("""
            **ℹ️ Note:** This is an academic demonstration tool created for a deep learning coursework project. 
            The model achieves 87.77% test accuracy on held-out data. Always consult a professional mycologist 
            before consuming any wild mushrooms. Many species are highly poisonous and can be fatal.
            """)
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            st.error("Please ensure you upload a valid image file (JPG, JPEG, or PNG).")
    
    else:
        # Display information when no file is uploaded
        st.info("👆 Please upload a mushroom image to begin classification")
        
        # Display example of supported classes
        with st.expander("📋 Supported Mushroom Species (9 Genera)"):
            st.markdown("""
            The model can identify the following mushroom genera:
            """)
            
            cols = st.columns(3)
            for idx, class_name in enumerate(CLASS_NAMES):
                with cols[idx % 3]:
                    st.markdown(f"**{idx+1}.** {class_name}")
            
            st.markdown("---")
            st.markdown("""
            **Performance by class:**
            - Best: Hygrocybe (97.92%), Boletus (96.89%)
            - Good: Amanita, Lactarius, Cortinarius (85-89%)
            - Challenging: Suillus (70.83%) - limited training data
            """)

# Run the application
if __name__ == "__main__":
    main()