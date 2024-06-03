from roboflow import Roboflow
import numpy as np
import supervision as sv
import cv2

PROJECT_NAME = "vehicle-on-road-car-colour"
VIDEO_FILE = "test2.mp4"  
ANNOTATED_VIDEO = "annotated2.mp4"   #What name to save file with

rf = Roboflow(api_key="")  #Change API Key
project = rf.workspace().project("vehicle-on-road-car-colour")
model = project.version("1").model

job_id, signed_url, expire_time = model.predict_video(
    VIDEO_FILE,
    fps=30,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)
print(results)


#box_mask_annotator = sv.BoxMaskAnnotator()
box_mask_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
tracker = sv.ByteTrack()
# box_annotator = sv.BoundingBoxAnnotator()
# halo_annotator = sv.HaloAnnotator()
# corner_annotator = sv.BoxCornerAnnotator()
# circle_annotator = sv.CircleAnnotator()
# blur_annotator = sv.BlurAnnotator()
# heat_map_annotator = sv.HeatMapAnnotator()

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
        #detections = tracker.track(detections)
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

# Assuming UTF-8 encoding, change to something else if you need to


with open(ANNOTATED_VIDEO, 'rb') as video_file:
    video_data = base64.b64encode(video_file.read()).decode()

print("Done")