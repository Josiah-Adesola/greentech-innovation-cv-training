from ultralytics import YOLO
import shutil

# Load your trained model
model = YOLO("/home/josiah/greentech/best.pt")

# Export to OpenVINO format
export_dir = model.export(format="openvino")

# Zip the exported model for download
shutil.make_archive("openvino_model", "zip", export_dir)

# Download the zipped model
files.download("openvino_model.zip")
