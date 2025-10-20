import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2
from PIL import Image

st.set_page_config(page_title="TrashNet Detector", layout="wide")

st.title("♻️ Trash Classification and Detection App")
st.markdown("Upload an **image or video** to detect waste materials using your trained YOLO model.")

# Load model (change to your path)
model_path = "best.pt"  # or "best.onnx" if you converted it
model = YOLO(model_path)

# Sidebar for mode selection
option = st.sidebar.selectbox("Select input type:", ("Image", "Video"))

if option == "Image":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        # Save temporary image
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_img.read())
            img_path = tmp.name

        # Predict
        results = model(img_path)
        annotated_frame = results[0].plot()  # Draw bounding boxes

        # Display result
        st.image(annotated_frame, caption="Detected Image", use_container_width=True)

elif option == "Video":
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_vid is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_vid.read())
            vid_path = tmp.name

        st.info("⏳ Processing video... please wait.")
        results = model.track(source=vid_path, save=True)

        # Get output video
        output_dir = results[0].save_dir
        video_files = [f for f in os.listdir(output_dir) if f.endswith((".mp4", ".avi"))]
        if video_files:
            out_video_path = os.path.join(output_dir, video_files[0])
            st.video(out_video_path)
        else:
            st.error("No output video found.")

st.success("✅ Ready for inference!")
