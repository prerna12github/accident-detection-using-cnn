import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os, tempfile, time

# --- Constants ---
IMG_HEIGHT, IMG_WIDTH = 250, 250
FRAME_SKIP = 30
MAX_ACCIDENT_FRAMES = 30
BORDER_SIZE = 10

# --- Streamlit Config ---
st.set_page_config(
    page_title="Accident Prediction App",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Model Loader ---
@st.cache_resource
def load_prediction_model():
    model_path = 'accident_prerna.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Prediction Logic ---
def predict_frame(model, frame_array):
    try:
        resized = tf.keras.preprocessing.image.smart_resize(frame_array, (IMG_HEIGHT, IMG_WIDTH))
        img = tf.keras.utils.img_to_array(resized)
        img_batch = np.expand_dims(img, axis=0)
        pred = model.predict(img_batch)[0][0]
        return "Accident Detected" if pred < 0.5 else "No Accident"
    except:
        return "Prediction Error"

# --- UI Header ---
st.title("🚨 Real-time Accident Prediction from Video")
st.write("Upload a video file to analyze for potential accidents.")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

model = load_prediction_model()

# --- Process Video ---
if uploaded_file and model:
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video.")
        os.unlink(video_path)
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    st.write(f"Total Frames: {total_frames}, FPS: {fps:.2f}, Frame Skip: {FRAME_SKIP}")

    # Placeholders
    frame_ph, pred_ph = st.empty(), st.empty()
    prog_bar, status_ph = st.progress(0), st.empty()

    frame_count = accident_count = 0
    accident_frames = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = predict_frame(model, rgb)

            text = f"Frame {frame_count}: {prediction}"
            color = (255, 0, 0) if prediction == "Accident Detected" else (0, 255, 0)
            caption = "ACCIDENT DETECTED!" if prediction == "Accident Detected" else "Current Analysis"

            if prediction == "Accident Detected":
                accident_count += 1
                if len(accident_frames) < MAX_ACCIDENT_FRAMES:
                    thumbnail = cv2.resize(rgb, (400, int(400 * rgb.shape[0] / rgb.shape[1])))
                    accident_frames.append((frame_count, thumbnail))
                rgb = cv2.copyMakeBorder(rgb, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                                         cv2.BORDER_CONSTANT, value=(255, 0, 0))

            cv2.putText(rgb, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            frame_ph.image(rgb, channels="RGB", use_container_width=True, caption=caption)
            pred_ph.markdown(f"**Current Prediction: {prediction}**")

            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed if elapsed else 0
            status_ph.text(f"Processed: {frame_count}, Accidents: {accident_count}, Speed: {fps_proc:.1f} FPS")

        prog_bar.progress(min(100, int((frame_count / total_frames) * 100)))

    cap.release()
    os.unlink(video_path)

    # --- Summary ---
    st.success("Video analysis complete!")
    st.subheader("Analysis Summary")
    st.write(f"Frames analyzed: {frame_count // FRAME_SKIP}")
    st.write(f"Accidents detected: {accident_count}")

    if accident_frames:
        st.subheader(f"🚨 Accident Frames Detected ({len(accident_frames)})")
        for i in range(0, len(accident_frames), 4):
            cols = st.columns(4)
            for j, (fnum, img) in enumerate(accident_frames[i:i+4]):
                cols[j].image(img, caption=f"Frame {fnum}", use_container_width=True)
    else:
        st.success("✅ No accident frames detected.")

# --- Sidebar Info ---
with st.sidebar:
    st.header("About")
    st.info("This application uses deep learning to detect road accidents in videos.")
