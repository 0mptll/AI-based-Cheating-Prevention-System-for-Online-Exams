# from ultralytics import YOLO
# import cv2

# class ObjectDetector:
#     def __init__(self, model_path='yolov8n.pt', device='cpu'):
#         self.model = YOLO(model_path)
#         self.device = device
#         self.unauthorized_labels = ['cell phone', 'book']  # add more as needed

#     def detect_unauthorized_objects(self, frame):
#         results = self.model.predict(source=frame, device=0 if self.device == 'cuda' else 'cpu', verbose=False)
#         detections = results[0].boxes
#         unauthorized_objects = []

#         for i in range(len(detections)):
#             cls_id = int(detections.cls[i])
#             conf = float(detections.conf[i])
#             xyxy = detections.xyxy[i].cpu().numpy().astype(int)
#             label = self.model.names[cls_id]

#             if label in self.unauthorized_labels:
#                 unauthorized_objects.append((
#                     xyxy[0], xyxy[1], xyxy[2], xyxy[3], label, conf
#                 ))

#         return unauthorized_objects