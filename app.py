import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from models import StudentModel
import os
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SECTORS = 12

# Set page config
st.set_page_config(
    page_title="Sutures Tension Prediction",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Slit-lamp Sutures Tension Prediction")
st.markdown("Upload a slit-lamp image to predict high-tension regions")

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = Path(__file__).parent / "student_angular_model.pth"
    model = StudentModel(NUM_SECTORS)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_array):
    """Preprocess image for model inference"""
    # Convert PIL Image to numpy array if needed
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array)
    
    # Ensure RGB format
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Already RGB
        pass
    
    # Resize to model input size
    processed = cv2.resize(image_array, (224, 224))
    
    # Normalize
    processed = processed.astype(np.float32) / 255.0
    
    # Convert to tensor
    processed = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)
    
    return processed.to(DEVICE)

def predict(image_array):
    """Run inference on the image"""
    model = load_model()
    x = preprocess_image(image_array)
    
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]
    
    return pred

# Create two columns for input and output
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a slit-lamp image...",
        type=["jpg", "jpeg", "png", "bmp"]
    )

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    with col1:
        st.image(image, use_column_width=True, caption="Uploaded Image")
    
    # Run inference
    try:
        with st.spinner("Running inference..."):
            predictions = predict(image_array)
        
        with col2:
            st.subheader("📊 Prediction Results (Sorted by Score)")
            
            # Display predictions as a table sorted by score in descending order
            results_data = []
            for i, score in enumerate(predictions):
                hour = f"{i}-{i+1} o'clock" if i < 11 else "11-12 o'clock"
                results_data.append({
                    "Sector": i + 1,
                    "Position": hour,
                    "Score": score
                })
            
            # Sort by score in descending order
            import pandas as pd
            df_results = pd.DataFrame(results_data)
            df_results = df_results.sort_values(by="Score", ascending=False).reset_index(drop=True)
            df_results["Score"] = df_results["Score"].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            # Display as bar chart sorted by score
            # st.subheader("📈 Scores Visualization (Sorted)")
            # chart_data = {
            #     "Sector": [f"S{i+1}" for i in range(len(predictions))],
            #     "Score": predictions
            # }
            
            # df_chart = pd.DataFrame(chart_data)
            # df_chart = df_chart.sort_values(by="Score", ascending=False).reset_index(drop=True)
            # st.bar_chart(df_chart.set_index("Sector"))
            
            # Summary statistics
            st.subheader("📋 Summary")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Max Score", f"{np.max(predictions):.4f}")
            
            with col_b:
                st.metric("Min Score", f"{np.min(predictions):.4f}")
            
            with col_c:
                st.metric("Mean Score", f"{np.mean(predictions):.4f}")
    
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        st.error("Please make sure the image is a valid slit-lamp image.")

else:
    with col2:
        st.info("👈 Upload an image to see predictions")
