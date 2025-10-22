# 🌱 Greentech Innovation — Computer Vision Training

This repository contains the training files and resources for the **Greentech Innovation Challenge** computer vision project.  
The model was trained for real-time object detection and optimized for deployment on edge devices.

---

## 📁 Project Overview
This project demonstrates:
- Training a custom object detection model using **YOLOv11**
- Exporting the model for **TensorFlow Lite** deployment
- Integration with **Streamlit** for real-time inference on CPU devices

---

## 🧠 Model Files
| File | Description | Link |
|------|--------------|------|
| **best.pt** | Trained YOLOv8 PyTorch model | [Download here](https://drive.google.com/file/d/1VgwTvhu8znl2flfcVVpUIIGRQsrVtbRT/view?usp=sharing) |
| **best_float32.tflite** | TensorFlow Lite version for mobile/edge deployment | [Download here](https://drive.google.com/file/d/19xpfDprjslCvwbONr41H8TM71D6BcoVq/view?usp=sharing) |

---

## 📓 Colab Notebook
You can view or rerun the full training process here:  
🔗 [Open in Google Colab](https://colab.research.google.com/drive/1E0bCAn0gqXa8UttyyNiGTtRxQA0H-hJ1)


This is the colab notebook to convert from the torch model to tflite32:  
🔗 [Open in Google Colab](https://colab.research.google.com/drive/1NuLfw659-xaLrXqdAExC0cv29MdHMOuB)
---

## ⚙️ Deployment Example
You can deploy this model using **Streamlit** for local inference.  
Example script structure:
```bash
streamlit run app.py
