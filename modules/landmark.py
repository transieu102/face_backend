import dlib
import cv2
class LandmarkDetector:
    def __init__(self):
        self.detector = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")
    def detect(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return self.detector(gray, dlib.rectangle(0, 0, gray.shape[1]-1, gray.shape[0]-1))
    def well_face(self, face_img):
            landmarks = self.detect(face_img)
            if len(landmarks.parts()) < 60:
                return False
            required_landmark_indices = [36, 45, 30]  # Left eye corner, right eye corner, and nose tip
            required_landmark_indices += list(range(48, 68))
            for index in required_landmark_indices:
                if not 0 <= landmarks.part(index).x < face_img.shape[1] or not 0 <= landmarks.part(index).y < face_img.shape[0]:
                    return False
            return True