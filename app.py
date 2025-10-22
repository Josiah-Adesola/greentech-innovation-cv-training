import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import yaml
import time
import os
from pathlib import Path

# -------------------------------
# Load class names
# -------------------------------
with open("data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml["names"]

# -------------------------------
# Initialize TFLite model
# -------------------------------
model_path = "best_float32.tflite"  # or "best_float16.tflite" or "best_int8.tflite"

if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' not found! Please ensure the TFLite model is in the same directory.")
    st.info("💡 Convert your model to TFLite format if you haven't already")
    st.stop()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.sidebar.subheader("🔍 Model Info")
st.sidebar.write(f"**Input shape:** {input_details[0]['shape']}")
st.sidebar.write(f"**Input dtype:** {input_details[0]['dtype']}")
st.sidebar.write(f"**Output shape:** {output_details[0]['shape']}")
st.sidebar.write(f"**Output dtype:** {output_details[0]['dtype']}")

# -------------------------------
# Helper function to run inference
# -------------------------------
def tflite_infer(image, confidence_threshold=0.25):
    img_bgr = image.copy()
    h, w = img_bgr.shape[:2]
    
    # Get input shape from model
    input_shape = input_details[0]['shape']
    input_height = input_shape[1]
    input_width = input_shape[2]
    
    # Preprocess image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_width, input_height))
    
    # Normalize to [0, 1]
    input_data = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Parse output based on shape
    boxes, scores, class_ids = parse_yolo_output(output_data, w, h, confidence_threshold)
    
    # Apply NMS
    if boxes:
        boxes, scores, class_ids = apply_nms(boxes, scores, class_ids, iou_threshold=0.45)
    
    return boxes, scores, class_ids

def parse_yolo_output(output, orig_w, orig_h, conf_thresh):
    """Parse YOLOv8 TFLite output"""
    boxes, scores, class_ids = [], [], []
    
    # Remove batch dimension if present
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    
    # YOLOv8 output format: (num_classes + 4, num_detections) or (num_detections, num_classes + 4)
    # Check and transpose if needed
    if output.shape[0] < output.shape[1] and output.shape[0] <= 10:
        output = output.T  # Transpose to (num_detections, num_classes + 4)
    
    # Format: [cx, cy, w, h, class0_conf, class1_conf, ..., class5_conf]
    boxes_data = output[:, :4]  # [cx, cy, w, h]
    class_scores = output[:, 4:]  # [class0, class1, ..., class5]
    
    for i in range(len(boxes_data)):
        # Get the class with highest score
        cls_id = int(np.argmax(class_scores[i]))
        conf = class_scores[i][cls_id]
        
        if conf > conf_thresh:
            cx, cy, w_box, h_box = boxes_data[i]
            
            # Convert from normalized/center format to pixel corner format
            # Check if coordinates are normalized (0-1) or in pixels (0-640)
            if cx <= 1.0 and cy <= 1.0:
                # Normalized coordinates
                x1 = int((cx - w_box / 2) * orig_w)
                y1 = int((cy - h_box / 2) * orig_h)
                w_scaled = int(w_box * orig_w)
                h_scaled = int(h_box * orig_h)
            else:
                # Pixel coordinates (need to scale from 640x640 to original)
                x1 = int((cx - w_box / 2) * orig_w / 640)
                y1 = int((cy - h_box / 2) * orig_h / 640)
                w_scaled = int(w_box * orig_w / 640)
                h_scaled = int(h_box * orig_h / 640)
            
            # Clip to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            w_scaled = min(w_scaled, orig_w - x1)
            h_scaled = min(h_scaled, orig_h - y1)
            
            boxes.append([x1, y1, w_scaled, h_scaled])
            scores.append(float(conf))
            class_ids.append(cls_id)
    
    return boxes, scores, class_ids

def apply_nms(boxes, scores, class_ids, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to remove overlapping boxes"""
    indices = cv2.dnn.NMSBoxes(
        boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        return (
            [boxes[i] for i in indices],
            [scores[i] for i in indices],
            [class_ids[i] for i in indices]
        )
    return [], [], []

def draw_detections(image, boxes, scores, class_ids):
    """Draw bounding boxes on image"""
    img_display = image.copy()
    
    # Color map for different classes
    colors = {
        'cardboard': (139, 69, 19),    # Brown
        'glass': (0, 255, 255),        # Cyan
        'metal': (192, 192, 192),      # Silver
        'paper': (255, 255, 255),      # White
        'plastic': (255, 0, 255),      # Magenta
        'trash': (128, 128, 128),      # Gray
    }
    
    for i, box in enumerate(boxes):
        x, y, w, h = box
        cls_id = class_ids[i]
        label = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        color = colors.get(label, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 3)
        
        # Draw label background
        label_text = f"{label}: {scores[i]:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_display, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        
        # Draw label text
        cv2.putText(img_display, label_text, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img_display

def display_results(img_display, boxes, scores, class_ids, elapsed):
    """Display detection results"""
    st.success(f"✅ Found {len(boxes)} objects in {elapsed:.3f}s")
    st.image(img_display, channels="BGR", caption="Detection Results", width='stretch')
    
    # Create summary by class
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Detection Summary")
        class_counts = {}
        for cls_id in class_ids:
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            st.metric(label=class_name.capitalize(), value=count)
    
    with col2:
        st.subheader("🎯 Average Confidence")
        for class_name in sorted(set(class_names[cid] for cid in class_ids)):
            class_scores = [scores[i] for i, cid in enumerate(class_ids) 
                           if class_names[cid] == class_name]
            avg_conf = np.mean(class_scores)
            st.metric(label=class_name.capitalize(), value=f"{avg_conf:.2%}")
    
    # Detailed detections
    with st.expander("🔍 View Detailed Detections"):
        detections = [{
            "id": i + 1,
            "class": class_names[cid] if cid < len(class_names) else str(cid),
            "confidence": f"{scores[i]:.4f}",
            "box": {"x": boxes[i][0], "y": boxes[i][1], "width": boxes[i][2], "height": boxes[i][3]}
        } for i, cid in enumerate(class_ids)]
        st.json(detections)

def process_video(video_path, confidence_threshold, progress_bar, status_text):
    """Process video file and return annotated video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_stats = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        boxes, scores, class_ids = tflite_infer(frame, confidence_threshold)
        
        # Draw detections
        frame_display = draw_detections(frame, boxes, scores, class_ids)
        
        # Write frame
        out.write(frame_display)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames} - Detected {len(boxes)} objects")
        
        detection_stats.append(len(boxes))
    
    cap.release()
    out.release()
    
    return output_path, detection_stats

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="TFLite Object Detection", layout="wide")
st.title("♻️ Waste Detection with TensorFlow Lite")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    confidence_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    nms_threshold = st.slider(
        "📦 NMS IoU Threshold", 
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="IoU threshold for Non-Maximum Suppression"
    )

# Main content - Input source selection
st.header("📥 Select Input Source")

input_source = st.radio(
    "Choose input type:",
    ["Upload Image", "Sample Images", "Camera", "Upload Video"],
    horizontal=True
)

img = None
video_file = None

# Upload Image
if input_source == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Sample Images
elif input_source == "Sample Images":
    sample_dir = Path("test/images")
    
    if sample_dir.exists():
        sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.jpeg")) + list(sample_dir.glob("*.png"))
        
        if sample_images:
            st.info(f"📁 Found {len(sample_images)} sample images")
            
            # Create columns for image selection
            cols = st.columns(4)
            selected_image = None
            
            for idx, img_path in enumerate(sample_images[:8]):  # Show first 8 images
                with cols[idx % 4]:
                    img_thumb = cv2.imread(str(img_path))
                    img_thumb = cv2.resize(img_thumb, (150, 150))
                    st.image(img_thumb, channels="BGR", caption=img_path.name, width='stretch')
                    if st.button(f"Select", key=f"btn_{idx}"):
                        selected_image = img_path
            
            # Dropdown for all images
            selected_path = st.selectbox(
                "Or select from dropdown:",
                options=sample_images,
                format_func=lambda x: x.name
            )
            
            if selected_image or selected_path:
                img_path = selected_image if selected_image else selected_path
                img = cv2.imread(str(img_path))
        else:
            st.warning("⚠️ No sample images found in test/images folder")
    else:
        st.warning("⚠️ test/images folder not found")

# Camera
elif input_source == "Camera":
    camera_image = st.camera_input("📷 Take a picture")
    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Upload Video
elif input_source == "Upload Video":
    video_file = st.file_uploader("🎬 Upload a video", type=["mp4", "avi", "mov", "mkv"])

# Display and process image
if img is not None:
    st.image(img, channels="BGR", caption="Input Image", width='stretch')

    if st.button("🚀 Run Detection", type="primary"):
        with st.spinner("Running TFLite inference..."):
            start_time = time.time()
            boxes, scores, class_ids = tflite_infer(img, confidence_threshold)
            elapsed = time.time() - start_time

        if not boxes:
            st.warning("⚠️ No objects detected.")
            st.info("💡 Try lowering the confidence threshold")
        else:
            img_display = draw_detections(img, boxes, scores, class_ids)
            display_results(img_display, boxes, scores, class_ids, elapsed)

# Process video
if video_file is not None:
    # Save uploaded video temporarily
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    
    st.video(temp_video_path)
    
    if st.button("🎬 Process Video", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Processing video..."):
            output_path, detection_stats = process_video(
                temp_video_path, 
                confidence_threshold, 
                progress_bar, 
                status_text
            )
        
        st.success("✅ Video processing complete!")
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Frames", len(detection_stats))
            st.metric("Avg Detections per Frame", f"{np.mean(detection_stats):.2f}")
        with col2:
            st.metric("Max Detections in Frame", max(detection_stats))
            st.metric("Frames with Detections", sum(1 for d in detection_stats if d > 0))
        
        # Show processed video
        st.video(output_path)
        
        # Download button
        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Processed Video",
                data=file,
                file_name="detected_video.mp4",
                mime="video/mp4"
            )
        
        # Cleanup
        os.remove(temp_video_path)