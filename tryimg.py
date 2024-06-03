from ultralytics import YOLO
import cv2
from roboflow import Roboflow
rf = Roboflow(api_key="")  #Change API Key
project = rf.workspace().project("vehicle-on-road-car-colour")
model = project.version("1").model
model.predict("cars.jpg", confidence=40, overlap=30).save("prediction.jpg")


#model1 = YOLO(model)
# infer on a local image
#print(model.predict("your_image.jpg", confidence=40, overlap=30).json())
# visualize your prediction
#results = model1.predict(source="0",show=True)
#print(results)
# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())


