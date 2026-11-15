import cv2
from pathlib import Path

class FaceDetector:
    def __init__(self, cascade_path=None):
        project_root = Path(__file__).resolve().parents[1]
        cascade_path = Path(cascade_path) if cascade_path else (
            project_root / "models" / "haarcascade_frontalface_default.xml"
        )
        if not cascade_path.exists():
            raise FileNotFoundError(
                f"Haar Cascade file not found: {cascade_path}. "
                "Place haarcascade_frontalface_default.xml in models/."
            )
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError("Could not load Haar Cascade classifier.")

    def detect_faces(self, frame):
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return self.detector.detectMultiScale(
            gray, scaleFactor=1.03, minNeighbors=3,
            minSize=(25, 25), flags=cv2.CASCADE_SCALE_IMAGE
        )
