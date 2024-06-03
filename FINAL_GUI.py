from ultralytics import YOLO
import cv2
from roboflow import Roboflow
import numpy as np
import supervision as sv

def image_Call():
    image_path=openFile1.filepath
    rf = Roboflow(api_key="5yAnDGYsrc6xaSUYr46m")  #Change API Key
    project = rf.workspace().project("vehicle-on-road-car-colour")
    model = project.version("1").model
    model.predict(image_path, confidence=40, overlap=30).save("prediction.jpg")

def video_call():
    PROJECT_NAME = "vehicle-on-road-car-colour"
    VIDEO_FILE = openFile2.filepath        #testing file we have to upload
    ANNOTATED_VIDEO = "annotated2.mp4"   #What name to save file with

    rf = Roboflow(api_key="5yAnDGYsrc6xaSUYr46m")  #Change API Key
    project = rf.workspace().project("vehicle-on-road-car-colour")
    model = project.version("1").model

    job_id, signed_url, expire_time = model.predict_video(
        VIDEO_FILE,
        fps=30,
        prediction_type="batch-video",
    )

    results = model.poll_until_video_results(job_id)
    print(results)

    box_mask_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    tracker = sv.ByteTrack()


    cap = cv2.VideoCapture(VIDEO_FILE)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    def annotate_frame(frame: np.ndarray, frame_number: int) -> np.ndarray:
        try:
            time_offset = frame_number / frame_rate
            closest_time_offset = min(results['time_offset'], key=lambda t: abs(t - time_offset))
            index = results['time_offset'].index(closest_time_offset)
            detection_data = results[PROJECT_NAME][index]

            roboflow_format = {
                "predictions": detection_data['predictions'],
                "image": {"width": frame.shape[1], "height": frame.shape[0]}
            }
            detections = sv.Detections.from_roboflow(roboflow_format)
            detections = tracker.update_with_detections(detections)
            labels = [pred['class'] for pred in detection_data['predictions']]

        except (IndexError, KeyError, ValueError) as e:
            print(f"Exception in processing frame {frame_number}: {e}")
            detections = sv.Detections(xyxy=np.empty((0, 4)),
                                    confidence=np.empty(0),
                                    class_id=np.empty(0, dtype=int))
            labels = []

        annotated_frame = box_mask_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        return annotated_frame

    sv.process_video(
        source_path=VIDEO_FILE,
        target_path=ANNOTATED_VIDEO,
        callback=annotate_frame
    )

    import base64

    with open(ANNOTATED_VIDEO, 'rb') as video_file:
        video_data = base64.b64encode(video_file.read()).decode()

    print("Done")

#####                                           CODE FOR GUI
#####                                              STARTS    
#####                                               NOW    

import customtkinter as cs
import tkinter as tk
import tkinter.messagebox
import tkinterDnD
from tkinter import filedialog as fd

cs.set_ctk_parent_class(tkinterDnD.Tk)
cs.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
cs.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

root=cs.CTk()
root.geometry("1000x500")
root.title("CAR COLOR PREDICTOR")

def openFile1():
    f_types1=[('Jpg files','*.jpg'),('PNG files','*.png')]
    openFile1.filepath = fd.askopenfilename(filetypes=f_types1)
    label_1.configure(text=(openFile1.filepath))

def openFile2():
    f_types2=[('MP4 files','*.mp4')]
    openFile2.filepath = fd.askopenfilename(filetypes=f_types2)
    label_3.configure(text=(openFile2.filepath))

frame=cs.CTkFrame(master=root)
frame.pack(pady=20,padx=60, fill="both", expand=True)

label = cs.CTkLabel(master=frame, text="Predicting of car color", justify=cs.LEFT)
label.pack(pady=10, padx=10)

tabview_1 = cs.CTkTabview(master=frame, width=400, height=250,)
tabview_1.pack(pady=10, padx=10)

tab1=tabview_1.add("Image Choosing")
button1=cs.CTkButton(master=tab1, text="Browse", command=openFile1)
button1.pack(pady=12, padx=10)
label_1 = cs.CTkLabel(master=tab1,text="",justify=cs.LEFT)
label_1.pack(pady=10, padx=10)

tab2=tabview_1.add("IMAGE PREDICTION")
button2b=cs.CTkButton(master=tab2, text="PREDICT", command=image_Call)
button2b.pack(pady=12, padx=10)

tab3=tabview_1.add("Video Choosing")
button3=cs.CTkButton(master=tab3, text="UPLOAD", command=openFile2)
button3.pack(pady=12, padx=10)
label_3 = cs.CTkLabel(master=tab3,text="",justify=cs.LEFT)
label_3.pack(pady=10, padx=10)

tab4=tabview_1.add("VIDEO PREDICTION")
button4b=cs.CTkButton(master=tab4, text="DETECT", command=video_call)
button4b.pack(pady=12, padx=10)

root.mainloop()