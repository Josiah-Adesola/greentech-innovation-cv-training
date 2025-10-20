import cv2
import torch
from super_gradients.training import models

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = models.get("yolo_nas_s", pretraeined_weights="coco").to(device)

out = model.predict("images/beautiful-cabin.jpeg", conf=0.4)
out.show()

##video
out = model.predict('videos/josiah.mp4', conf=0.8)

## webcam
out = model.predict_webcam()


