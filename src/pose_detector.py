"""Un mic ajutor peste MediaPipe Pose, ca restul scripturilor să rămână scurte.

Folosire:
    detector = DetectorCorp()
    puncte = detector.detecteaza(cadru)   # lista cu 33 de puncte sau None
"""
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

CALE_MODEL = "models/pose_landmarker_full.task"


class DetectorCorp:
    def __init__(self, video=True, segmentare=False):
        """
        video=True      -> pentru camera live (folosește și cadrele anterioare, e mai stabil)
        video=False     -> pentru imagini independente (Ziua 8)
        segmentare=True -> calculează și masca persoanei (Ziua 9)
        """
        self.video = video
        optiuni = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=CALE_MODEL),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            output_segmentation_masks=segmentare,
        )
        self.detector = vision.PoseLandmarker.create_from_options(optiuni)
        self.start = time.monotonic()
        self.ultimul_timp_ms = -1
        self.ultima_masca = None

    def detecteaza(self, cadru_bgr):
        """Primește un cadru de la cameră și returnează cele 33 de puncte (sau None)."""
        # Lab 7 / Lab 13: OpenCV ține culorile în ordinea BGR, MediaPipe vrea RGB
        rgb = cv2.cvtColor(cadru_bgr, cv2.COLOR_BGR2RGB)
        imagine = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.video:
            # În modul video, MediaPipe vrea un timp (în ms) care crește mereu
            timp_ms = max(int((time.monotonic() - self.start) * 1000), self.ultimul_timp_ms + 1)
            self.ultimul_timp_ms = timp_ms
            rezultat = self.detector.detect_for_video(imagine, timp_ms)
        else:
            rezultat = self.detector.detect(imagine)

        self.ultima_masca = None
        if rezultat.segmentation_masks:
            self.ultima_masca = np.squeeze(rezultat.segmentation_masks[0].numpy_view())

        if not rezultat.pose_landmarks:
            return None
        return rezultat.pose_landmarks[0]   # prima (și singura) persoană
