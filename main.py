import cv2
from modules.detector import Detector
from modules.features_extractor import Extractor
detector = Detector()
extractor = Extractor()
cap = cv2.VideoCapture(0)  
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    if not ret:
        break
    face = detector.detect(frame)
    if face[0] is not None:
        x, y, x2, y2 = [int(i) for i in face[0][0]]
        face_img = frame[y:y2, x:x2]
        vector = extractor.extract(face_img)
        print(vector.shape)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Facial Landmark Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
