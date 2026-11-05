import csv
import os
import sys

import cv2

from features import NUME_CARACTERISTICI, extrage_caracteristici
from pose_detector import DetectorCorp

# Rulare: python src/collect_data.py numele_tau acasa_zi
if len(sys.argv) != 3:
    print("Folosire: python src/collect_data.py <persoana> <sesiune>")
    print("Exemplu:  python src/collect_data.py valentin acasa_zi")
    sys.exit(1)

persoana, sesiune = sys.argv[1], sys.argv[2]
print(f"Persoana: {persoana}, sesiunea: {sesiune}")
TASTE = {ord("1"): "drept", ord("2"): "cocosat", ord("3"): "aplecat"}


os.makedirs("data", exist_ok=True)
cale_csv = f"data/{persoana}_{sesiune}.csv"

# Nu suprascriem o sesiune deja înregistrată
if os.path.exists(cale_csv):
    print(f"Exista deja {cale_csv}. Alege alt nume de sesiune.")
    sys.exit(1)

fisier = open(cale_csv, "w", newline="")
scriitor = csv.writer(fisier)
scriitor.writerow(["persoana", "sesiune", "eticheta"] + NUME_CARACTERISTICI)
print(f"Salvez datele in {cale_csv}")

detector = DetectorCorp()
camera = cv2.VideoCapture(0)
eticheta = None   # None = pauză, nu salvăm nimic
numar = {"drept": 0, "cocosat": 0, "aplecat": 0}


while True:
    ok, cadru = camera.read()
    if not ok:
        print("Nu pot citi de la camera.")
        break

    puncte = detector.detecteaza(cadru)
        
    if eticheta and puncte:
        caracteristici = extrage_caracteristici(puncte)
        if caracteristici:
            scriitor.writerow([persoana, sesiune, eticheta] + caracteristici)
            numar[eticheta] += 1
            
            # Păstrăm 1 imagine din 15 rânduri salvate, pentru experimentul din Ziua 8
            if numar[eticheta] % 15 == 0:
                folder = f"data/imagini/{eticheta}"
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(f"{folder}/{persoana}_{sesiune}_{numar[eticheta]}.jpg", cadru)



    afisare = cv2.flip(cadru, 1)
    text = f"Eticheta: {eticheta or 'PAUZA'}   {numar}"
    cv2.putText(afisare, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Colectare date", afisare)

    tasta = cv2.waitKey(1) & 0xFF
    if tasta in TASTE:
        eticheta = TASTE[tasta]
    elif tasta == ord("0"):
        eticheta = None
    elif tasta == ord("q"):
        break


fisier.close()
camera.release()
cv2.destroyAllWindows()
print("Gata.")

