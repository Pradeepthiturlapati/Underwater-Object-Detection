import streamlit as st
import numpy as np
import cv2
import tempfile
import time
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import os
# Import constants from the utils folder
from utils.constants import (
    MODEL_FILE_PATH, IMG_SIZE, CLASS_NAMES, 
    FINAL_METRICS, ACCURACY_CURVE_PATH, LOSS_CURVE_PATH,
    SAMPLE_IMAGE_PATH, SAMPLE_VIDEO_PATH
)

# --- CONFIGURATION & STYLING ---

st.set_page_config(
    page_title="Underwater Object Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for a better look (underwater theme)
st.markdown("""
<style>
    .reportview-container .main {
        background: linear-gradient(135deg, #0f3057, #1b659c);
        color: #f0f2f6;
    }
    h1, h2, h3, .st-bh, .st-ds, .st-em, .st-en, .st-es {
        color: #87CEEB !important; /* Sky Blue for headers */
    }
    .stButton>button {
        background-color: #279EFF;
        color: white;
        border-radius: 12px;
        border: 2px solid #0077b6;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0077b6;
        border-color: #279EFF;
    }
    .metric-value {
        color: #FFD700 !important; /* Gold for metrics */
    }
    .sidebar .sidebar-content {
        background-color: #0f3057;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. MODEL LOADING AND CACHING ---
@st.cache_resource
def load_and_cache_model():
    """Loads the Keras model using Streamlit's cache resource."""
    if not os.path.exists(MODEL_FILE_PATH):
        st.error(f"Model file not found at: {MODEL_FILE_PATH}")
        return None
        
    try:
        model = load_model(MODEL_FILE_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model. Check TensorFlow/Keras version compatibility. Error: {e}")
        return None

hybrid_model = load_and_cache_model()

# --- 2. PREDICTION FUNCTIONS ---

def predict_image(img_path, model):
    """Predicts the class of a single uploaded image file."""
    try:
        # Load and preprocess image using PIL
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        pred = model.predict(img_array, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred)
        
        return CLASS_NAMES[pred_class], confidence
    except Exception as e:
        st.error(f"Error during image prediction: {e}")
        return None, None

def save_video_source_to_temp(video_source):
    """Saves the video source (UploadedFile or path string) to a temporary file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            # Check if the source is an UploadedFile object (has .read() method)
            if hasattr(video_source, 'read'):
                tmp_file.write(video_source.read())
            # Check if the source is a string path (for sample files)
            elif isinstance(video_source, str):
                # Read the sample file content in binary mode
                with open(video_source, 'rb') as f:
                    tmp_file.write(f.read())
            else:
                st.error("Invalid video source provided.")
                return None
            
            return tmp_file.name
    except FileNotFoundError:
        st.error(f"Sample video file not found.")
        return None


# --- 3. PAGE FUNCTIONS ---

def model_test_page():
    """Renders the model testing interface for images and videos."""
    st.title("🌊 Hybrid Model Underwater Object Testing")
    st.markdown("""
        Upload an image or a video, or use the sample files, to test the 
        **Hybrid Underwater Object Classification Model** (32 classes).
    """)
    
    if hybrid_model is None:
        return

    st.markdown("---")

    # Sample File Selector
    col_sample, col_upload = st.columns([1, 2])
    
    with col_sample:
        st.subheader("1. Sample Files")
        sample_option = st.radio(
            "Select Sample File:", 
            ["None", "Sample Image", "Sample Video"], 
            index=0,
            key="sample_radio"
        )
        
    with col_upload:
        st.subheader("2. Upload File")
        uploaded_file = st.file_uploader(
            "Upload Image (.jpg, .png) or Video (.mp4)", 
            type=['jpg', 'png', 'mp4'], 
            accept_multiple_files=False,
            key="main_uploader"
        )
        
    # Determine the source (uploaded or sample)
    source_file = None
    if uploaded_file is not None:
        source_file = uploaded_file
    elif sample_option == "Sample Image":
        source_file = SAMPLE_IMAGE_PATH
    elif sample_option == "Sample Video":
        source_file = SAMPLE_VIDEO_PATH

    st.markdown("---")

    if source_file is not None:
        # Determine file extension based on source type
        if hasattr(source_file, 'name'):
            file_extension = os.path.splitext(source_file.name)[1].lower()
        elif isinstance(source_file, str):
            file_extension = os.path.splitext(source_file)[1].lower()
        else:
            file_extension = ""
            
        
        if file_extension in ['.jpg', '.png']:
            # --- IMAGE PREDICTION ---
            st.subheader("Image Classification Result")
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.image(source_file, caption=f"Input: {os.path.basename(source_file.name if hasattr(source_file, 'name') else source_file)}", use_container_width=True)
            
            with col2:
                with st.spinner('Analyzing image...'):
                    input_to_predictor = source_file
                    label, confidence = predict_image(input_to_predictor, hybrid_model)
                
                if label:
                    st.success("Analysis Complete!")
                    st.metric(label="Predicted Class", value=label, delta="High Confidence" if confidence > 0.9 else None, delta_color="normal")
                    st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
                else:
                    st.error("Failed to classify the image.")

        elif file_extension in ['.mp4']:
            # --- VIDEO LIVE ANALYSIS ---
            st.subheader("Video Live Analysis")
            
            # Display the original video for reference
            st.markdown("#### Original Video (For Reference)")
            st.video(source_file, format='video/mp4')

            if st.button("Start Live Video Analysis", key="start_analysis_btn"):
                
                # 1. Save source to temp file path
                temp_video_path = save_video_source_to_temp(source_file)
                if temp_video_path is None:
                    return

                # 2. Setup video capture
                cap = cv2.VideoCapture(temp_video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                st.markdown("#### Processed Video Output (Live Feed)")
                
                # Create placeholders for live video and progress
                video_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                start_time = time.time()
                current_frame = 0
                
                st.info("Analysis in progress...")

                # 3. Live Processing Loop
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Preprocess frame for model
                    img = cv2.resize(frame, IMG_SIZE)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_input = img_rgb.astype("float32") / 255.0
                    img_input = np.expand_dims(img_input, axis=0)

                    # Predict class
                    preds = hybrid_model.predict(img_input, verbose=0) 
                    class_id = np.argmax(preds)
                    label = CLASS_NAMES[class_id]

                    # Draw prediction on frame (original size)
                    cv2.putText(frame, f"Predicted: {label}", 
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.1, (255, 255, 255), 3, cv2.LINE_AA)
                    
                    # Display the processed frame live using the placeholder
                    # We convert BGR (OpenCV) back to RGB for Streamlit/Image
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    
                    # Update progress bar
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    progress_bar.progress(int(current_frame / frame_count * 100))

                # 4. Cleanup
                cap.release()
                os.unlink(temp_video_path)
                
                progress_bar.progress(100) # Ensure it hits 100%
                st.success(f"Live Analysis complete! Total duration: {time.time() - start_time:.2f} seconds.")
                    
        else:
            st.warning(f"Unsupported file type: {file_extension}. Please use .jpg, .png, or .mp4.")


def model_metrics_page():
    """Renders the training and validation metrics."""
    st.title("📈 Training & Validation Metrics")
    st.markdown("---")

    st.header("Final Model Performance")
    st.markdown(f"""
        The Hybrid Model achieved an exceptional **Best Validation Accuracy** of **{FINAL_METRICS['Best Validation Accuracy']:.4f}** over the training period.
    """)
    
    col_acc, col_loss = st.columns(2)
    
    with col_acc:
        st.metric(label="Best Validation Accuracy", value=f"{FINAL_METRICS['Best Validation Accuracy']:.4f}", delta="High generalization", delta_color="normal")
    
    with col_loss:
        st.metric(label="Final Training Loss", value=f"{FINAL_METRICS['Final Training Loss']:.4f}", delta="Low loss", delta_color="inverse")
    
    st.markdown("---")

    # Training Plots
    st.header("Model Convergence Curves (15 Epochs)")
    st.markdown("Visual representation of the model's learning stability.")
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.subheader("Accuracy Curve")
        if os.path.exists(ACCURACY_CURVE_PATH):
            st.image(ACCURACY_CURVE_PATH, caption="Accuracy Curve (Train vs. Validation)", use_container_width=True)
            st.info("The curves closely track each other, indicating the model generalizes well with minimal overfitting.")
        else:
            st.warning(f"Accuracy curve image not found at: {ACCURACY_CURVE_PATH}")

    with col_plot2:
        st.subheader("Loss Curve")
        if os.path.exists(LOSS_CURVE_PATH):
            st.image(LOSS_CURVE_PATH, caption="Loss Curve (Train vs. Validation)", use_container_width=True)
            st.info("Both training and validation loss dropped quickly and remained stable, demonstrating effective optimization.")
        else:
            st.warning(f"Loss curve image not found at: {LOSS_CURVE_PATH}")


# --- 4. NAVIGATION AND EXECUTION ---

st.sidebar.title("Navigation")
st.sidebar.markdown("### Underwater Model")

if hybrid_model is None:
    st.sidebar.error("Model Error")
    st.stop()
else:
    page = st.sidebar.radio("Explore:", ["Model Test", "Model Metrics"])

if page == "Model Test":
    model_test_page()
elif page == "Model Metrics":
    model_metrics_page()