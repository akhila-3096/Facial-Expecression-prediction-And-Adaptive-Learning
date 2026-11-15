import cv2
from src.face_detection import FaceDetector
from src.emotion_prediction import EmotionPredictor

def run_camera(camera_index=0):
    detector = FaceDetector()
    predictor = EmotionPredictor()
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            faces = detector.detect_faces(frame)
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                emotion, confidence = predictor.predict(frame[y:y+h, x:x+w])
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{emotion}: {confidence:.1%}",
                            (x, max(25,y-10)), cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (0,255,0), 2)
            cv2.imshow("NeuroVision AI - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
