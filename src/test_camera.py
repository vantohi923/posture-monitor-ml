import cv2
from mediapipe.tasks.python.vision import PoseLandmarksConnections, drawing_utils

from pose_detector import DetectorCorp

detector = DetectorCorp()
camera = cv2.VideoCapture(0)   # 0 = camera laptopului

while True:
    ok, cadru = camera.read()
    if not ok:
        print("Nu pot citi de la camera. E folosita de alta aplicatie?")
        break

    puncte = detector.detecteaza(cadru)

    # Desenăm punctele și liniile dintre ele (scheletul)
    if puncte:
        drawing_utils.draw_landmarks(cadru, puncte, PoseLandmarksConnections.POSE_LANDMARKS)

    # Afișăm în oglindă, ca să fie natural pentru utilizator
    cv2.imshow("Test camera (q = iesire)", cv2.flip(cadru, 1))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
