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


def _create_sector_masks(h, w, num_sectors=12, center=None, radius=None):
    """Create boolean masks for each sector (clockwise)"""
    ys, xs = np.ogrid[:h, :w]
    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center
    dx = xs - cx
    dy = ys - cy
    # distance from center
    dist = np.sqrt(dx * dx + dy * dy)
    if radius is None:
        radius = 0.95 * min(cx, cy)

    # angle in degrees, range [0,360). We flip y so 0 degrees is to the right and increases counter-clockwise
    angles = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360

    masks = []
    sector_angle = 360.0 / num_sectors
    for i in range(num_sectors):
        start = i * sector_angle
        end = (i + 1) * sector_angle
        mask = (angles >= start) & (angles < end) & (dist <= radius)
        masks.append(mask)

    return masks


def make_overlay(image_array, scores, alpha=0.35, colormap=cv2.COLORMAP_JET):
    """Return an RGB image (PIL) with a colored sector overlay based on `scores`.

    image_array must be an HxWx3 RGB uint8 numpy array.
    """
    def _pentacam_lut():
        # Anchors roughly matching Pentacam axial-like colors: deep blue -> cyan -> green -> yellow -> red
        anchors = np.array([0, 64, 128, 192, 255], dtype=np.float32)
        colors = np.array([
            [0, 0, 128],    # deep blue
            [0, 255, 255],  # cyan
            [0, 255, 0],    # green
            [255, 255, 0],  # yellow
            [255, 0, 0],    # red
        ], dtype=np.float32)
        xs = np.arange(256, dtype=np.float32)
        lut = np.zeros((256, 3), dtype=np.uint8)
        for c in range(3):
            lut[:, c] = np.interp(xs, anchors, colors[:, c]).astype(np.uint8)
        return lut

    pentacam_lut = None
    if isinstance(colormap, str) and colormap.upper().startswith("PENTA"):
        pentacam_lut = _pentacam_lut()
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array)

    h, w = image_array.shape[:2]

    # normalize scores to 0-255
    scores_np = np.array(scores, dtype=np.float32)
    smin = scores_np.min()
    smax = scores_np.max()
    if smax - smin < 1e-6:
        norm = np.zeros_like(scores_np)
    else:
        norm = (scores_np - smin) / (smax - smin)
    vals = (norm * 255).astype(np.uint8)

    # prepare overlay in BGR (for cv2 colormap) then convert to RGB
    overlay_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    masks = _create_sector_masks(h, w, num_sectors=len(scores_np))
    for i, mask in enumerate(masks):
        val = int(vals[i])
        if pentacam_lut is not None:
            # pentacam_lut is RGB; convert to BGR for overlay_bgr
            rgb = pentacam_lut[val]
            color = np.array([int(rgb[2]), int(rgb[1]), int(rgb[0])], dtype=np.uint8)
        else:
            color = cv2.applyColorMap(np.full((1, 1), val, dtype=np.uint8), colormap)[0, 0]
        # fill overlay where mask is True
        overlay_bgr[mask] = color

    # Convert overlay to RGB
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    orig = image_array.astype(np.float32)
    over = overlay_rgb.astype(np.float32)
    blended = (over * alpha + orig * (1.0 - alpha)).astype(np.uint8)

    # Where overlay is zero (no sector region), show original
    mask_any = (overlay_bgr.sum(axis=2) > 0)
    blended[~mask_any] = orig[~mask_any].astype(np.uint8)

    return Image.fromarray(blended)

# Create two columns for input and output
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a slit-lamp image...",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    # Overlay controls
    show_overlay = st.checkbox("Show heatmap overlay", value=True)
    alpha_val = st.slider("Overlay transparency", 0.0, 1.0, 0.35, step=0.05)
    cmap_choice = st.selectbox("Colormap", ["Jet", "Plasma", "Viridis", "Hot"], index=0)
    cmap_map = {
        "Jet": cv2.COLORMAP_JET,
        "Plasma": cv2.COLORMAP_PLASMA,
        "Viridis": cv2.COLORMAP_VIRIDIS,
        "Hot": cv2.COLORMAP_HOT,
        "Pentacam": "PENTACAM",
    }
    cmap = cmap_map.get(cmap_choice, cv2.COLORMAP_JET)

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
        
        # Show overlay in the left column if requested
        if show_overlay:
            try:
                overlay_img = make_overlay(image, predictions, alpha=alpha_val, colormap=cmap)
                with col1:
                    st.subheader("🗺️ Heatmap Overlay")
                    st.image(overlay_img, use_column_width=True, caption="Overlayed Image")
            except Exception as e:
                with col1:
                    st.error(f"Error creating overlay: {e}")
    
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        st.error("Please make sure the image is a valid slit-lamp image.")

else:
    with col2:
        st.info("👈 Upload an image to see predictions")
