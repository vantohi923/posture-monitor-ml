import cv2
import joblib
import pandas as pd

from features import NUME_CARACTERISTICI, extrage_caracteristici
from pose_detector import DetectorCorp

model = joblib.load("models/model.joblib")
detector = DetectorCorp()
camera = cv2.VideoCapture(0)

while True:
    ok, cadru = camera.read()
    if not ok:
        print("Nu pot citi de la camera.")
        break

    puncte = detector.detecteaza(cadru)
    postura = None

    if puncte:
        caracteristici = extrage_caracteristici(puncte)
        if caracteristici:
            intrare = pd.DataFrame([caracteristici], columns=NUME_CARACTERISTICI)
            postura = model.predict(intrare)[0]

    afisare = cv2.flip(cadru, 1)
    cv2.putText(afisare, f"Postura: {postura or '-'}", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    cv2.imshow("Monitor postura", afisare)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
