def well_defined_face( landmarks, image ) -> bool:
    if len(landmarks.parts()) < 60:
        return False
    required_landmark_indices = [36, 45, 30]  # Left eye corner, right eye corner, and nose tip
    required_landmark_indices += list(range(48, 68))
    for index in required_landmark_indices:
        if not 0 <= landmarks.part(index).x < image.shape[1] or not 0 <= landmarks.part(index).y < image.shape[0]:
            return False
    return True
