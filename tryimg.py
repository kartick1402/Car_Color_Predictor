from ultralytics import YOLO
import cv2
from roboflow import Roboflow
rf = Roboflow(api_key="")  #Change API Key
project = rf.workspace().project("vehicle-on-road-car-colour")
model = project.version("1").model
model.predict("cars.jpg", confidence=40, overlap=30).save("prediction.jpg")
