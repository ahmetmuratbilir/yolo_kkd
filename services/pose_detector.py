# ============================================================
#  services/pose_detector.py  –  MediaPipe Pose Landmark Takibi
# ============================================================
import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class PoseDetector:
    """
    MediaPipe Pose kütüphanesini kullanarak insan iskelet yapısını tespit eder
    ve sol/sağ bilek koordinatlarını bulur.
    """

    def __init__(self):
        self.enabled = MEDIAPIPE_AVAILABLE
        if self.enabled:
            print("[PoseDetector] MediaPipe Pose yukleniyor...")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            print("[PoseDetector] UYARI: 'mediapipe' yuklu degil. Pose takibi pasif (fallback aktif).")

    # ------------------------------------------------------------------ #
    def find_wrists(self, frame) -> list[dict]:
        """
        Kare üzerindeki tüm bilek koordinatlarını döner.
        Çıktı formatı:
            [
                {
                    "left_wrist": (x, y) veya None,
                    "right_wrist": (x, y) veya None
                },
                ...
            ]
        """
        if not self.enabled:
            return []

        h, w = frame.shape[:2]
        # MediaPipe RGB formatında çalışır
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        wrists = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Sol Bilek (LEFT_WRIST = 15)
            left_w = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            left_coord = None
            if left_w.visibility > 0.5:
                left_coord = (int(left_w.x * w), int(left_w.y * h))

            # Sağ Bilek (RIGHT_WRIST = 16)
            right_w = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            right_coord = None
            if right_w.visibility > 0.5:
                right_coord = (int(right_w.x * w), int(right_w.y * h))

            wrists.append({
                "left_wrist": left_coord,
                "right_wrist": right_coord
            })

        return wrists
