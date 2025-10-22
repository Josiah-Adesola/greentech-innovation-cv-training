"""
Run this code in Google Colab for optimal results.


"""


from ultralytics import YOLO

# Load your model
model = YOLO('/content/drive/MyDrive/best.pt')

# Export to TFLite (float32)
model.export(format='tflite', imgsz=640)

print("✅ Conversion complete! Look for 'best_saved_model' folder and 'best_float32.tflite' file")

from google.colab import files

files.download('/content/best_saved_model/best_float32.tflite')