import mediapipe as mp
import cv2
import time
import numpy as np


mp_face_detection = mp.solutions.face_detection


class FaceGuard:
    def __init__(self):
        self.detector = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def analyze(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)

        if not results.detections:
            return "NO_FACE", 0

        if len(results.detections) > 1:
            return "MULTIPLE_FACES", len(results.detections)

        return "OK", 1
