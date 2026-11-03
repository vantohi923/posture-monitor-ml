import cv2
from mediapipe.tasks.python.vision import PoseLandmarksConnections, drawing_utils

from features import NUME_CARACTERISTICI, extrage_caracteristici
from pose_detector import DetectorCorp

detector = DetectorCorp()
camera = cv2.VideoCapture(0)   # 0 = camera laptopului

while True:
    ok, cadru = camera.read()
    if not ok:
        print("Nu pot citi de la camera. E folosita de alta aplicatie?")
        break

    puncte = detector.detecteaza(cadru)
    caracteristici = None

    # Desenăm punctele și liniile dintre ele (scheletul)
    if puncte:
        drawing_utils.draw_landmarks(cadru, puncte, PoseLandmarksConnections.POSE_LANDMARKS)
        caracteristici = extrage_caracteristici(puncte)

    # Afișăm în oglindă, ca să fie natural pentru utilizator
    afisare = cv2.flip(cadru, 1)

    # Scriem numerele după flip, altfel textul ar apărea în oglindă
    if caracteristici:
        for i, (nume, valoare) in enumerate(zip(NUME_CARACTERISTICI, caracteristici)):
            cv2.putText(afisare, f"{nume}: {valoare:+.2f}", (10, 25 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(afisare, "Umerii nu se vad", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Test camera (q = iesire)", afisare)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
