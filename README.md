# 🪑 Monitor de postură cu Machine Learning

> O aplicație care te urmărește prin camera laptopului și te anunță când stai cocoșat.
> Modelul ML e antrenat pe date strânse de mine, iar tehnicile de la cursul de
> **Prelucrarea Imaginilor** sunt folosite ca să meargă bine și la lumină slabă.

**Stadiu:** ✅ Ziua 1–3 (detecție, caracteristici, colectare date) · 🔄 Ziua 4 (explorare gata, prima sesiune: ~7.500 rânduri echilibrate; urmează colegii) · ⏳ Ziua 5–10

---

## 📌 Cuprins

1. [Ce problemă rezolvă](#-ce-problemă-rezolvă)
2. [Cum funcționează (pe scurt)](#-cum-funcționează-pe-scurt)
3. [Rezultatul final](#-rezultatul-final)
4. [Structura proiectului](#-structura-proiectului)
5. [Legătura cu laboratoarele de Prelucrarea Imaginilor](#-legătura-cu-laboratoarele-de-prelucrarea-imaginilor)
6. [Planul pe 10 zile](#-planul-pe-10-zile)
7. [Dicționar ML (cuvintele pe care trebuie să le știi)](#-dicționar-ml)
8. [Întrebări de interviu la care trebuie să știi să răspunzi](#-întrebări-de-interviu)

---

## ❓ Ce problemă rezolvă

Studenții și programatorii stau 6–10 ore pe zi la calculator. Fără să ne dăm seama,
ne cocoșăm sau ne apropiem prea mult de ecran, ceea ce duce la dureri de spate și de gât.

**Aplicația:**
- rulează în timp ce lucrezi, folosind doar camera laptopului;
- recunoaște 3 posturi: **drept**, **cocoșat**, **aplecat spre ecran**;
- te avertizează doar dacă stai greșit **mai mult de 10 secunde** (nu la fiecare mișcare);
- îți arată statistici: „azi ai stat drept 72% din timp”.

---

## ⚙️ Cum funcționează (pe scurt)

```
 Camera ──► Preprocesare ──► MediaPipe Pose ──► Caracteristici ──► Model ML ──► Netezire ──► Alertă
 (cadru)    (CLAHE, Lab 3,13)  (33 puncte ale    (unghiuri,          (Random     în timp      + statistici
                                corpului)          distanțe)           Forest)    (Lab 4, 5)
```

1. **Camera** dă un cadru (o imagine).
2. **Preprocesare:** dacă e întuneric, îmbunătățim contrastul (tehnici din laborator).
3. **MediaPipe Pose** (un model gata antrenat de Google) găsește 33 de puncte ale corpului: nas, urechi, umeri etc.
4. **Caracteristici:** din puncte calculăm numere care descriu postura, de exemplu
   „cât de sus e capul față de umeri”.
5. **Modelul ML, antrenat de mine,** primește aceste numere și spune: drept / cocoșat / aplecat.
6. **Netezire în timp:** nu ne luăm după un singur cadru, ci după ultimele câteva (filtre din laborator).
7. **Alertă:** dacă postura greșită durează peste 10 secunde, ecranul devine roșu.

> **Ce e ML-ul meu aici?** MediaPipe doar găsește punctele. **Decizia** (ce postură e)
> o ia modelul pe care **l-am antrenat eu**, pe date **strânse de mine**, evaluat corect.

---

## 🏁 Rezultatul final

La sfârșitul celor 10 zile trebuie să ai:

| # | Ce | Unde |
|---|---|---|
| 1 | Aplicație live care detectează postura și dă alerte | `src/app.py` |
| 2 | Set de date propriu (CSV), de la tine și 3–5 colegi | `data/` |
| 3 | Comparația a 3 modele, cu evaluare **pe persoane noi** | `src/train.py` |
| 4 | Matricea de confuzie și importanța caracteristicilor | `results/*.png` |
| 5 | Experimentul de robustețe (întuneric, zgomot, blur) cu și fără preprocesare | `results/robustete.png` |
| 6 | Graficul cu progresul tău zilnic | `results/progres.png` |
| 7 | Cod pe GitHub, cu README, plus video de 45 de secunde pe LinkedIn | — |

Exemplu de tabel final (cifrele tale vor fi altele):

| Model | F1 (împărțire aleatorie) ❌ | F1 (persoane noi) ✅ |
|---|---|---|
| Baseline (ghicește mereu clasa cea mai frecventă) | 0.17 | 0.17 |
| Regresie logistică | 0.91 | 0.82 |
| Random Forest | 0.99 | 0.86 |

> Observă: la împărțirea aleatorie rezultatele sunt **prea bune ca să fie adevărate**.
> Explicăm de ce în Ziua 5. E cea mai importantă lecție din proiect.

---

## 📁 Structura proiectului

```
ProiectML/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── pose_detector.py    # Ziua 1 – MediaPipe Pose într-o clasă mică (refolosită peste tot)
│   ├── test_camera.py      # Ziua 1 – verificăm camera și MediaPipe
│   ├── features.py         # Ziua 2 – calculul caracteristicilor
│   ├── collect_data.py     # Ziua 3 – strângem date (apăsând taste)
│   ├── explore.py          # Ziua 4 – ne uităm la date
│   ├── train.py            # Ziua 5 – antrenăm și comparăm modele
│   ├── evaluate.py         # Ziua 6 – analiza erorilor, salvăm modelul
│   ├── app.py              # Ziua 7 – aplicația live
│   ├── preprocesare.py     # Ziua 8 – tehnicile din laboratoare
│   ├── robustness.py       # Ziua 8 – experimentul de robustețe
│   └── stats.py            # Ziua 9 – statistici
├── data/                   # CSV-uri + imagini/ (tot folderul rămâne local, NU se urcă pe GitHub)
├── models/                 # modelul MediaPipe (.task, descărcat) + modelul nostru salvat
├── sessions/               # jurnalul de utilizare (pentru statistici)
└── results/                # grafice pentru README și LinkedIn
```

**Regulă:** toate scripturile se rulează **din folderul principal** `ProiectML`, de exemplu:
```bash
python src/test_camera.py
```

---

## 🎓 Legătura cu laboratoarele de Prelucrarea Imaginilor

La laborator am lucrat în C++ cu OpenCV. Aici folosim **aceleași funcții**, dar în Python:
`cv::GaussianBlur(...)` devine `cv2.GaussianBlur(...)`.

| Laborator | Ce am învățat | Unde folosesc în proiect | Ziua |
|---|---|---|---|
| **Lab 7, Lab 13**: spații de culoare | `cvtColor`, RGB, HSI, L\*a\*b\* | Camera dă **BGR**, MediaPipe vrea **RGB** | 1 |
| **Lab 4**: filtrul medie | media pixelilor vecini | **Media în timp** a predicțiilor (ultimele 15 cadre) | 7 |
| **Lab 5**: filtrul median | elimină zgomotul „sare și piper” | **Mediana în timp** a caracteristicilor (elimină săriturile bruște ale punctelor) | 7 |
| **Lab 2**: corecția gamma | `s = c · r^γ` | Simulăm **întunericul** în experimentul de robustețe | 8 |
| **Lab 4**: filtrul gaussian | netezire, blur | Simulăm o cameră **nefocalizată** și reducem zgomotul | 8 |
| **Lab 3**: egalizarea adaptivă a histogramei | contrast mai bun local | **CLAHE** pentru lumină slabă | 8 |
| **Lab 13**: egalizare pe luminanță, nu pe R, G, B | egalizarea pe componente strică culorile | CLAHE aplicat doar pe canalul **L** din L\*a\*b\* | 8 |
| **Lab 10**: binarizare și operații morfologice | `threshold`, `morphologyEx` (închidere) | **Încețoșarea fundalului** pentru video (confidențialitate) | 9 |
| **Lab 4, Lab 6**: convoluția, Sobel | filtre alese de mână | Explicația pentru CNN la interviu: un CNN **învață singur** filtrele | — |

---

## 📅 Planul pe 10 zile

Fiecare zi durează aproximativ **2–3 ore**. Fiecare zi are:
- 🎯 **Scopul** zilei
- 🛠️ **Pașii**
- ✅ **Gata când** (cum știi că ai terminat)
- 🧠 **Ce ai învățat** (pentru interviu)

---

### Ziua 1 – Instalare și primul test cu camera

🎯 **Scop:** vezi scheletul tău desenat live pe ecran.

🛠️ **Pași:**

1. Creează un mediu virtual (izolează bibliotecile acestui proiect).
   Folosim Python-ul **oficial** (3.13), prin `py -3.13`. Comanda simplă `python` poate porni
   alt Python (de ex. din MSYS2), pentru care MediaPipe nu are pachete.
   ```bash
   cd E:\An3\ProiectML
   py -3.13 -m venv venv
   venv\Scripts\activate
   ```
2. Instalează bibliotecile din `requirements.txt` (versiunile exacte sunt fixate acolo):
   ```bash
   pip install -r requirements.txt
   ```
3. Descarcă modelul MediaPipe Pose (aproximativ 9 MB, de la Google):
   ```bash
   curl -L -o models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
   ```
4. `src/pose_detector.py`: ascunde partea de pregătire a MediaPipe ca să nu o repetăm în fiecare script.
   Îl folosești în două linii:
   ```python
   detector = DetectorCorp()
   puncte = detector.detecteaza(cadru)   # lista cu 33 de puncte sau None
   ```
   În interior: convertește cadrul din **BGR în RGB** (Lab 7 / Lab 13), îl dă modelului și
   returnează punctele primei persoane.

   > MediaPipe 1.x a scos vechiul API `mp.solutions.pose` (pe care îl vezi în multe tutoriale).
   > Aici folosim API-ul nou, `PoseLandmarker` („Tasks”), care are nevoie de fișierul `.task` de la pasul 3.

5. `src/test_camera.py`:

```python
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
```

6. Rulează: `python src/test_camera.py`

| Problemă | Soluție |
|---|---|
| `activate` dă eroare în PowerShell | Rulează o dată `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Camera nu pornește | Închide Teams, Zoom sau alte aplicații care folosesc camera. Sau încearcă `VideoCapture(1)`. |
| `No module named ...` | Nu ai activat mediul virtual: `venv\Scripts\activate` |

✅ **Gata când:** vezi scheletul pe corpul tău și se mișcă odată cu tine.

🧠 **Ce ai învățat:**
- MediaPipe Pose este un model **pre-antrenat**: primește o imagine și returnează 33 de puncte (x, y, z, vizibilitate).
- `x` și `y` sunt între 0 și 1 (procent din lățimea și înălțimea imaginii), iar `y` crește **în jos**.

---

### Ziua 2 – Caracteristicile (din puncte în numere)

🎯 **Scop:** transformi cele 33 de puncte în câteva numere care descriu postura.

**De ce?** Modelul nu „vede” poza. Îi dăm numere cu sens, de exemplu
„cât de sus e capul față de umeri”. Când te cocoșezi, capul coboară spre umeri, deci numărul scade.

**Truc important (normalizare):** dacă te apropii de cameră, totul pare mai mare.
De aceea **împărțim distanțele la lățimea umerilor**: numerele nu mai depind de cât de departe stai.

🛠️ **Pași:** creează `src/features.py`:

```python
import math

# Numerele punctelor MediaPipe de care avem nevoie
NAS = 0
URECHE_STG, URECHE_DR = 7, 8
UMAR_STG, UMAR_DR = 11, 12

NUME_CARACTERISTICI = [
    "cap_umeri",         # cât de sus sunt urechile față de umeri (scade când te cocoșezi)
    "nas_umeri",         # cât de sus e nasul față de umeri
    "latime_umeri",      # cât de aproape ești de cameră (umeri mai lați = mai aproape)
    "diferenta_umeri",   # un umăr mai jos decât celălalt
    "diferenta_urechi",  # capul înclinat într-o parte
    "adancime_cap",      # cât de în față e capul față de umeri (z de la MediaPipe)
]


def distanta(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def extrage_caracteristici(puncte):
    """Primește cele 33 de puncte și returnează o listă de 6 numere (sau None)."""
    # Dacă umerii nu se văd bine, nu avem încredere în calcul
    if puncte[UMAR_STG].visibility < 0.5 or puncte[UMAR_DR].visibility < 0.5:
        return None

    latime_umeri = distanta(puncte[UMAR_STG], puncte[UMAR_DR])
    if latime_umeri < 1e-6:
        return None

    umeri_y = (puncte[UMAR_STG].y + puncte[UMAR_DR].y) / 2
    urechi_y = (puncte[URECHE_STG].y + puncte[URECHE_DR].y) / 2
    umeri_z = (puncte[UMAR_STG].z + puncte[UMAR_DR].z) / 2

    # Împărțim la lățimea umerilor => nu contează distanța față de cameră
    return [
        (umeri_y - urechi_y) / latime_umeri,
        (umeri_y - puncte[NAS].y) / latime_umeri,
        latime_umeri,
        (puncte[UMAR_STG].y - puncte[UMAR_DR].y) / latime_umeri,
        (puncte[URECHE_STG].y - puncte[URECHE_DR].y) / latime_umeri,
        (puncte[NAS].z - umeri_z) / latime_umeri,
    ]
```

**Test:** în `test_camera.py`, după `draw_landmarks`, adaugă:

```python
from features import extrage_caracteristici   # (sus, la importuri)
...
        c = extrage_caracteristici(puncte)
        if c:
            print([round(x, 2) for x in c])
```

Stai drept, apoi cocoșat, și urmărește cum se schimbă numerele.

✅ **Gata când:** vezi clar că `cap_umeri` scade când te cocoșezi și `latime_umeri` crește când te apropii de ecran.

🧠 **Ce ai învățat:**
- **Feature engineering:** alegi numere cu sens fizic din datele brute.
- **Normalizarea** face modelul să funcționeze indiferent de distanța față de cameră.

---

### Ziua 3 – Strângem date (setul de date propriu)

🎯 **Scop:** un fișier CSV cu câteva mii de rânduri, fiecare etichetat cu postura.

**Cum:** pornești scriptul și apeși o tastă ca să spui ce postură ai **acum**:
`1` = drept, `2` = cocoșat, `3` = aplecat spre ecran, `0` = pauză, `q` = ieșire.

🛠️ **Pași:** creează `src/collect_data.py`:

```python
import csv
import os
import sys

import cv2

from features import extrage_caracteristici, NUME_CARACTERISTICI
from pose_detector import DetectorCorp

# Rulare: python src/collect_data.py andrei acasa_zi
persoana, sesiune = sys.argv[1], sys.argv[2]

TASTE = {ord("1"): "drept", ord("2"): "cocosat", ord("3"): "aplecat"}
SALVEAZA_O_IMAGINE_LA = 15   # salvăm 1 imagine din 15 cadre (pentru Ziua 8)

os.makedirs("data", exist_ok=True)
fisier = open(f"data/{persoana}_{sesiune}.csv", "w", newline="")
scriitor = csv.writer(fisier)
scriitor.writerow(["persoana", "sesiune", "eticheta"] + NUME_CARACTERISTICI)

detector = DetectorCorp()
camera = cv2.VideoCapture(0)
eticheta = None   # None = pauză, nu salvăm nimic
numar = {"drept": 0, "cocosat": 0, "aplecat": 0}
cadru_nr = 0

while True:
    ok, cadru = camera.read()
    if not ok:
        break
    cadru_nr += 1

    puncte = detector.detecteaza(cadru)

    if eticheta and puncte:
        caracteristici = extrage_caracteristici(puncte)
        if caracteristici:
            scriitor.writerow([persoana, sesiune, eticheta] + caracteristici)
            numar[eticheta] += 1

            # Păstrăm și câteva imagini pentru experimentul de robustețe
            if cadru_nr % SALVEAZA_O_IMAGINE_LA == 0:
                folder = f"data/imagini/{eticheta}"
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(f"{folder}/{persoana}_{sesiune}_{cadru_nr}.jpg", cadru)

    afisare = cv2.flip(cadru, 1)
    text = f"Eticheta: {eticheta or 'PAUZA'}   {numar}"
    cv2.putText(afisare, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
```

**Cum înregistrezi (aproximativ 10 minute):**
1. `python src/collect_data.py numele_tau acasa_zi`
2. Apeși `1` și stai **drept** 2 minute. Mișcă-te natural: tastezi, te uiți în stânga și în dreapta.
3. Apeși `2` și stai **cocoșat** 2 minute. Include și „puțin cocoșat”, nu doar exagerat!
4. Apeși `3` și stai **aplecat spre ecran** 2 minute.
5. Apeși `q`.

> ⚠️ Folosește **alt nume de sesiune** de fiecare dată. Dacă fișierul există deja, scriptul refuză să pornească, ca să nu pierzi o înregistrare.

✅ **Gata când:** ai `data/numele_tau_acasa_zi.csv` cu aproximativ 1000+ rânduri pentru fiecare postură.

🧠 **Ce ai învățat:**
- **Etichetare:** fiecare exemplu are răspunsul corect atașat.
- Datele trebuie să conțină și **cazuri grele** (puțin cocoșat), altfel modelul învață doar extremele.

---

### Ziua 4 – Mai multe date și prima privire asupra lor

🎯 **Scop:** date variate (mai mulți oameni și condiții) și verificarea că sunt bune.

🛠️ **Pași:**

1. **Mai multe sesiuni pentru tine:** seara (lumină slabă), alt scaun, cu hanorac, laptopul mai aproape sau mai departe.
   ```bash
   python src/collect_data.py numele_tau acasa_seara
   python src/collect_data.py numele_tau camin_zi
   ```
2. **3–5 colegi**, fiecare câte 10 minute. Cere-le **acordul**: imaginile lor rămân doar la tine și **nu se urcă pe GitHub**.
3. Creează `src/explore.py`:

```python
import glob

import matplotlib.pyplot as plt
import pandas as pd

from features import NUME_CARACTERISTICI

date = pd.concat([pd.read_csv(f) for f in glob.glob("data/*.csv")], ignore_index=True)

print("Total rânduri:", len(date))
print("\nRânduri pe postură:\n", date["eticheta"].value_counts())
print("\nRânduri pe persoană:\n", date["persoana"].value_counts())

# Pentru fiecare caracteristică: cum diferă valorile între posturi?
date.boxplot(column=NUME_CARACTERISTICI, by="eticheta", layout=(2, 3), figsize=(12, 7))
plt.suptitle("Caracteristicile pe fiecare postură")
plt.tight_layout()
plt.show()
```

✅ **Gata când:**
- ai date de la **cel puțin 3 persoane** (ideal 5);
- clasele sunt aproximativ **echilibrate** (număr apropiat de rânduri);
- în boxplot vezi că măcar 2–3 caracteristici **diferă clar** între posturi.

🧠 **Ce ai învățat:**
- **Clase echilibrate:** dacă 90% din date ar fi „drept”, un model care spune mereu „drept” ar avea 90% acuratețe și ar fi inutil.
- **Boxplot:** arată rapid dacă o caracteristică separă clasele.

---

### Ziua 5 – Antrenăm și comparăm modele (cea mai importantă zi!)

🎯 **Scop:** compari 3 modele și înțelegi **de ce trebuie testat pe persoane noi**.

**Capcana:** cadrele consecutive sunt aproape identice. Dacă amesteci aleatoriu toate
rândurile în antrenare și test, modelul vede la test practic **aceleași imagini** ca la
antrenare și rezultatul e fals de bun. Corect este să testezi pe **o persoană pe care
modelul n-a văzut-o niciodată** (*Leave-One-Person-Out*):

```
Runda 1: antrenez pe [Ana, Bogdan, Cristi, Dan]  →  testez pe [Tu]
Runda 2: antrenez pe [Tu, Bogdan, Cristi, Dan]   →  testez pe [Ana]
...                                                   apoi facem media
```

🛠️ **Pași:** creează `src/train.py`:

```python
import glob

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import NUME_CARACTERISTICI

date = pd.concat([pd.read_csv(f) for f in glob.glob("data/*.csv")], ignore_index=True)
X = date[NUME_CARACTERISTICI]   # intrările (caracteristicile)
y = date["eticheta"]            # răspunsul corect

# Testăm pe persoane noi; dacă ai o singură persoană, măcar pe sesiuni noi
grupuri = date["persoana"] if date["persoana"].nunique() > 1 else date["sesiune"]

modele = {
    "Baseline (clasa cea mai frecventa)": DummyClassifier(strategy="most_frequent"),
    "Regresie logistica": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
}

print(f"{'Model':40s} {'Aleator (GRESIT)':>18s} {'Persoane noi (CORECT)':>22s}")
for nume, model in modele.items():
    gresit = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0),
                             scoring="f1_macro")
    corect = cross_val_score(model, X, y, cv=LeaveOneGroupOut(), groups=grupuri,
                             scoring="f1_macro")
    print(f"{nume:40s} {gresit.mean():18.2f} {corect.mean():22.2f}")
```

Rulează: `python src/train.py`

✅ **Gata când:**
- vezi tabelul cu cele 3 modele;
- vezi că varianta „aleator” dă rezultate **mai mari** decât „persoane noi” și poți explica de ce;
- modelele reale bat clar baseline-ul.

🧠 **Ce ai învățat:**
- **Baseline:** un model prost, de referință. Dacă nu-l bați, modelul tău nu învață nimic.
- **Regresia logistică** trasează granițe drepte între clase. **StandardScaler** aduce toate caracteristicile la aceeași scară.
- **Random Forest:** 200 de arbori de decizie care votează.
- **F1 macro:** media F1 pe toate clasele. E mai corect decât acuratețea când clasele nu sunt perfect echilibrate.
- **Data leakage** (scurgere de date): testul conține informații din antrenare. **Cea mai bună poveste pentru interviu.**

---

### Ziua 6 – Analiza erorilor și salvarea modelului

🎯 **Scop:** afli **unde și de ce** greșește modelul, apoi îl salvezi.

🛠️ **Pași:** creează `src/evaluate.py`:

```python
import glob
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from features import NUME_CARACTERISTICI

date = pd.concat([pd.read_csv(f) for f in glob.glob("data/*.csv")], ignore_index=True)
X = date[NUME_CARACTERISTICI]
y = date["eticheta"]
grupuri = date["persoana"] if date["persoana"].nunique() > 1 else date["sesiune"]

model = RandomForestClassifier(n_estimators=200, random_state=0)
os.makedirs("results", exist_ok=True)

# 1. Predicții pe persoane noi (fiecare persoană e prezisă de un model care n-a văzut-o)
predictii = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=grupuri)
print(classification_report(y, predictii))

# 2. Matricea de confuzie: ce postură e confundată cu care
ConfusionMatrixDisplay.from_predictions(y, predictii, cmap="Blues")
plt.title("Matricea de confuzie (testat pe persoane noi)")
plt.savefig("results/matrice_confuzie.png", dpi=150)
plt.close()

# 3. Pe cine greșește cel mai des?
date["corect"] = predictii == y
print("\nAcuratețe pe fiecare persoană:\n", date.groupby(grupuri)["corect"].mean().round(2))

# 4. Antrenăm modelul final pe TOATE datele și vedem ce caracteristici contează
model.fit(X, y)
importanta = pd.Series(model.feature_importances_, index=NUME_CARACTERISTICI).sort_values()
importanta.plot.barh(title="Importanța caracteristicilor")
plt.tight_layout()
plt.savefig("results/importanta.png", dpi=150)
plt.close()

# 5. Salvăm modelul ca să-l folosim în aplicație
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
print("\nModel salvat în models/model.joblib")
```

✅ **Gata când:**
- ai `results/matrice_confuzie.png` și `results/importanta.png`;
- poți spune în **2 propoziții** unde greșește modelul. De exemplu: „Confundă «puțin cocoșat» cu «drept».
  Pe Bogdan merge mai prost pentru că poartă ochelari și stă mai departe de cameră.”
- ai `models/model.joblib`.

🧠 **Ce ai învățat:**
- **Matricea de confuzie:** pe diagonală sunt răspunsurile corecte, în afara ei confuziile.
- **Precision:** din cele prezise „cocoșat”, câte chiar erau. **Recall:** din cele care erau „cocoșat”, câte am prins.
- **Analiza erorilor** te ajută să știi ce date să mai strângi.

---

### Ziua 7 – Aplicația live 🎉

🎯 **Scop:** aplicația funcționează în timp real, cu ecran verde sau roșu și alertă.

**Două filtre din laborator, aplicate în timp:**
- **Lab 5, filtrul median:** dacă MediaPipe greșește pe un singur cadru, punctul „sare”.
  Luăm **mediana ultimelor 7 cadre**, la fel cum filtrul median elimină pixelii „sare și piper”.
- **Lab 4, filtrul medie:** facem **media probabilităților pe ultimele 15 cadre**, ca să nu „pâlpâie” predicția.

🛠️ **Pași:** creează `src/app.py`:

```python
import time
from collections import deque

import cv2
import joblib
import numpy as np
import pandas as pd

from features import extrage_caracteristici, NUME_CARACTERISTICI
from pose_detector import DetectorCorp

SECUNDE_PANA_LA_ALERTA = 10

model = joblib.load("models/model.joblib")
detector = DetectorCorp()
camera = cv2.VideoCapture(0)

ultimele_caracteristici = deque(maxlen=7)    # Lab 5: filtru median în timp
ultimele_probabilitati = deque(maxlen=15)    # Lab 4: filtru medie în timp
inceput_postura_gresita = None

while True:
    ok, cadru = camera.read()
    if not ok:
        break

    puncte = detector.detecteaza(cadru)
    postura = None

    if puncte:
        caracteristici = extrage_caracteristici(puncte)
        if caracteristici:
            ultimele_caracteristici.append(caracteristici)
            netezite = np.median(ultimele_caracteristici, axis=0)

            intrare = pd.DataFrame([netezite], columns=NUME_CARACTERISTICI)
            ultimele_probabilitati.append(model.predict_proba(intrare)[0])
            medie = np.mean(ultimele_probabilitati, axis=0)
            postura = model.classes_[np.argmax(medie)]

    # Alertă doar dacă postura greșită DUREAZĂ
    if postura and postura != "drept":
        if inceput_postura_gresita is None:
            inceput_postura_gresita = time.time()
        durata = time.time() - inceput_postura_gresita
    else:
        inceput_postura_gresita = None
        durata = 0
    alerta = durata > SECUNDE_PANA_LA_ALERTA

    # Afișare: chenar verde sau roșu (culorile în OpenCV sunt BGR!)
    afisare = cv2.flip(cadru, 1)
    culoare = (0, 0, 255) if alerta else (0, 200, 0)
    inaltime, latime = afisare.shape[:2]
    cv2.rectangle(afisare, (0, 0), (latime - 1, inaltime - 1), culoare, 15)
    cv2.putText(afisare, f"{postura or '-'}  ({durata:.0f}s)", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, culoare, 2)
    if alerta:
        cv2.putText(afisare, "INDREAPTA-TE!", (25, inaltime - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, culoare, 3)

    cv2.imshow("Monitor postura", afisare)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
```

✅ **Gata când:** te cocoșezi, după 10 secunde ecranul devine roșu, iar când te îndrepți revine la verde.

**Experiment:** pune `maxlen=1` la ambele cozi (adică fără filtre) și vezi cum „pâlpâie” predicția. Diferența e un exemplu bun pentru video.

🧠 **Ce ai învățat:**
- **Probabilitățile modelului** (`predict_proba`) sunt mai utile decât un singur răspuns, pentru că le poți face media.
- Un model bun **nu e suficient**: aplicația are nevoie și de logică (netezire, prag de timp) ca să fie utilizabilă.

---

### Ziua 8 – Experimentul de robustețe (laboratoarele în acțiune)

🎯 **Scop:** demonstrezi cu cifre că tehnicile din laborator fac modelul să meargă în condiții proaste.

**Ideea:**
1. Luăm imaginile salvate în Ziua 3.
2. Le **stricăm**: întuneric (gamma, Lab 2), zgomot, blur (Lab 4).
3. Măsurăm cât scade acuratețea.
4. Aplicăm **corecții** (CLAHE, Lab 3 și Lab 13) și măsurăm cât recuperăm.

🛠️ **Pași:**

**1. Creează `src/preprocesare.py`** (toate tehnicile din laborator într-un singur loc):

```python
import cv2
import numpy as np


# ---------- Degradări (simulăm condiții proaste) ----------

def intuneric(img, gamma=3.0):
    """Lab 2 – corecția gamma: s = 255 * (r/255)^gamma. Cu gamma > 1 imaginea se întunecă."""
    tabel = np.array([255 * (i / 255) ** gamma for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, tabel)   # LUT = aplică tabelul pe fiecare pixel (operație punctuală)


def zgomot(img, intensitate=25):
    """Zgomot gaussian, ca la o cameră ieftină seara."""
    z = np.random.normal(0, intensitate, img.shape)
    return np.clip(img + z, 0, 255).astype(np.uint8)


def blur(img):
    """Lab 4 – filtru gaussian mare: simulează o cameră nefocalizată."""
    return cv2.GaussianBlur(img, (21, 21), 0)


# ---------- Corecții ----------

def clahe_luminanta(img):
    """Lab 3 (egalizare adaptivă) + Lab 13 (doar pe luminanță, nu pe R, G, B separat)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def reducere_zgomot(img):
    """Lab 4 – filtru gaussian mic: potrivit pentru zgomot gaussian."""
    return cv2.GaussianBlur(img, (5, 5), 0)
```

**2. Creează `src/robustness.py`:**

```python
import glob
import os
import random

import cv2
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from features import extrage_caracteristici, NUME_CARACTERISTICI
from pose_detector import DetectorCorp
from preprocesare import intuneric, zgomot, blur, clahe_luminanta, reducere_zgomot

model = joblib.load("models/model.joblib")
detector = DetectorCorp(video=False)   # imagini independente, nu video


def prezice(img):
    puncte = detector.detecteaza(img)
    if not puncte:
        return None   # nu a găsit persoana => considerăm greșit
    c = extrage_caracteristici(puncte)
    if c is None:
        return None
    return model.predict(pd.DataFrame([c], columns=NUME_CARACTERISTICI))[0]


conditii = {
    "original": lambda img: img,
    "întuneric": intuneric,
    "întuneric + CLAHE": lambda img: clahe_luminanta(intuneric(img)),
    "zgomot": zgomot,
    "zgomot + filtru gaussian": lambda img: reducere_zgomot(zgomot(img)),
    "blur": blur,
}

random.seed(0)
imagini = glob.glob("data/imagini/*/*.jpg")
imagini = random.sample(imagini, min(300, len(imagini)))   # 300 ca să nu dureze mult

rezultate = {}
for nume, transformare in conditii.items():
    corecte = 0
    for cale in imagini:
        eticheta_reala = os.path.basename(os.path.dirname(cale))   # numele folderului
        corecte += prezice(transformare(cv2.imread(cale))) == eticheta_reala
    rezultate[nume] = corecte / len(imagini)
    print(f"{nume:28s} acuratețe: {rezultate[nume]:.2f}")

pd.Series(rezultate).plot.bar(title="Robustețea modelului în condiții proaste", ylim=(0, 1))
plt.ylabel("Acuratețe")
plt.tight_layout()
plt.savefig("results/robustete.png", dpi=150)
```

**3. Salvează câte o imagine din fiecare condiție** (original, întuneric, întuneric + CLAHE) una lângă alta.
Arată foarte bine în video și în README:

```python
import cv2, numpy as np
from preprocesare import intuneric, clahe_luminanta
img = cv2.imread("data/imagini/drept/<o_imagine>.jpg")
cv2.imwrite("results/comparatie.jpg", np.hstack([img, intuneric(img), clahe_luminanta(intuneric(img))]))
```

**4. Folosește CLAHE și în aplicație.** În `app.py`, chiar după `camera.read()`:

```python
from preprocesare import clahe_luminanta   # (sus)
...
    cadru = clahe_luminanta(cadru)
```

✅ **Gata când:** ai `results/robustete.png` și o frază de tipul:
„La întuneric, acuratețea scade de la 0.88 la 0.52. Cu CLAHE pe canalul L revine la 0.81.”

> Notă onestă: modelul a fost antrenat pe toate datele, deci și pe aceste imagini.
> Ce măsurăm aici e **diferența** dintre condiții, nu acuratețea absolută.

🧠 **Ce ai învățat:**
- **Robustețea:** un model bun în laborator poate cădea în condiții reale.
- De ce CLAHE pe **L** și nu pe R, G, B: egalizarea pe fiecare canal schimbă raportul dintre culori și apar culori false.
- **Augmentarea datelor:** aceleași funcții de degradare se pot folosi ca să generezi date noi de antrenare.

---

### Ziua 9 – Statistici și pregătirea pentru video

🎯 **Scop:** aplicația îți spune cât de bine ai stat, iar fundalul e ascuns pentru video.

🛠️ **Pași:**

**1. Jurnal de utilizare.** În `app.py`, salvează postura o dată pe secundă:

```python
import csv, datetime, os   # (sus)

os.makedirs("sessions", exist_ok=True)
cale_jurnal = f"sessions/{datetime.date.today()}.csv"
fisier_nou = not os.path.exists(cale_jurnal)
jurnal = open(cale_jurnal, "a", newline="")
scriitor = csv.writer(jurnal)
if fisier_nou:
    scriitor.writerow(["timp", "postura"])
ultima_salvare = 0

# ... în bucla while, după ce afli postura:
    if postura and time.time() - ultima_salvare >= 1:
        scriitor.writerow([datetime.datetime.now().isoformat(timespec="seconds"), postura])
        ultima_salvare = time.time()

# ... la final, lângă camera.release():
jurnal.close()
```

**2. Creează `src/stats.py`:**

```python
import glob

import matplotlib.pyplot as plt
import pandas as pd

date = pd.concat([pd.read_csv(f) for f in glob.glob("sessions/*.csv")], ignore_index=True)
date["zi"] = pd.to_datetime(date["timp"]).dt.date

procent_drept = date.groupby("zi")["postura"].apply(lambda p: (p == "drept").mean() * 100)
print(procent_drept.round(1))

procent_drept.plot.bar(title="Cât din timp am stat drept", ylim=(0, 100))
plt.ylabel("% din timp")
plt.tight_layout()
plt.savefig("results/progres.png", dpi=150)
```

**3. Încețoșarea fundalului pentru video (Lab 10: binarizare și morfologie).**
MediaPipe poate da și o **mască**: cât de probabil e ca fiecare pixel să aparțină persoanei.

```python
detector = DetectorCorp(segmentare=True)   # înlocuiește linia veche

# ... în buclă, după detector.detecteaza(cadru):
    if detector.ultima_masca is not None:
        masca = (detector.ultima_masca > 0.5).astype(np.uint8)            # binarizare (Lab 10)
        nucleu = np.ones((15, 15), np.uint8)
        masca = cv2.morphologyEx(masca, cv2.MORPH_CLOSE, nucleu)          # închidere (Lab 10)
        fundal = cv2.GaussianBlur(cadru, (55, 55), 0)                     # blur (Lab 4)
        cadru = np.where(masca[:, :, None] == 1, cadru, fundal)           # persoana clară, fundalul blurat
```

> Pune codul **după** calculul posturii și **înainte** de afișare, ca blurul să nu afecteze predicția.

4. **Folosește aplicația** câteva ore în timp ce înveți sau lucrezi, ca să ai date reale pentru `progres.png`.

✅ **Gata când:** ai `results/progres.png`, iar în aplicație fundalul e încețoșat și tu ești clar.

🧠 **Ce ai învățat:**
- Diferența dintre un model și un **produs** (jurnal, statistici, confidențialitate).
- **Operația de închidere** (dilatare + eroziune) umple găurile mici din mască.

---

### Ziua 10 – GitHub, video și LinkedIn

🎯 **Scop:** proiectul e public, arată profesionist și ai postat video-ul.

🛠️ **Pași:**

**1. Creează `.gitignore`** (ce NU se urcă pe GitHub):
```
venv/
data/
sessions/
__pycache__/
```
> Datele tale și ale colegilor (CSV-uri și imagini) **nu se urcă niciodată**: tot folderul `data/` rămâne doar pe laptop.

**2. Urcă pe GitHub:**
```bash
git init
git add .
git commit -m "Monitor de postura cu ML"
```
Apoi creează un repository pe github.com și urmează instrucțiunile de acolo (`git remote add ...`, `git push`).

**3. Actualizează README-ul** pentru vizitatori: un GIF cu aplicația sus de tot, tabelul cu modelele,
graficele din `results/` și secțiunea „Ce am învățat”.

**4. Video (aproximativ 45 de secunde, fără să depinzi de sunet):**

| Secunde | Ce arăți |
|---|---|
| 0–3 | Te cocoșezi, ecranul devine roșu și apare „ÎNDREAPTĂ-TE!” |
| 3–12 | Te îndrepți și ecranul devine verde. Scheletul desenat pe corp. |
| 12–22 | Pipeline-ul: cameră → puncte → caracteristici → model → alertă |
| 22–32 | Imaginea întuneric / CLAHE și graficul de robustețe |
| 32–45 | Tabelul cu modelele, graficul cu progresul tău și linkul spre GitHub |

**5. Postarea pe LinkedIn (șablon):**
> Stăteam cocoșat la laptop 60% din zi. Așa că am construit un model ML care mă supraveghează. 🪑
>
> 🔹 Am strâns propriul set de date: X cadre de la 5 persoane
> 🔹 Am comparat 3 modele; Random Forest: F1 = 0.86 pe persoane pe care modelul nu le-a văzut
> 🔹 Am aflat că testarea aleatorie dădea un fals 0.99 (data leakage!)
> 🔹 Am aplicat tehnici de la cursul de Prelucrarea Imaginilor (CLAHE, filtre median/medie)
>   ca să meargă și seara: acuratețea la lumină slabă a crescut de la 0.52 la 0.81
>
> După 2 săptămâni de folosire: de la 40% la 70% din timp cu postură corectă.
> Cod: [link GitHub]
> #MachineLearning #ComputerVision #Python

✅ **Gata când:** repository-ul e public, video-ul e postat și poți răspunde la toate întrebările de mai jos.

---

## 📖 Dicționar ML

| Termen | Pe înțelesul tuturor |
|---|---|
| **Caracteristică (feature)** | Un număr care descrie ceva despre exemplu (de ex. `cap_umeri`) |
| **Etichetă (label)** | Răspunsul corect (drept / cocoșat / aplecat) |
| **Antrenare** | Modelul vede mii de exemple cu răspunsul corect și învață singur regulile |
| **Clasificator** | Un model care alege o categorie |
| **Baseline** | Un model foarte simplu, de comparație. Trebuie să-l bați. |
| **Overfitting** | Modelul învață pe de rost datele de antrenare și merge prost pe date noi |
| **Data leakage** | Informații din test „scapă” în antrenare, deci rezultate false |
| **Cross-validation** | Împarți datele în mai multe feluri, antrenezi și testezi de mai multe ori, apoi faci media |
| **Leave-One-Person-Out** | La fiecare rundă, o persoană e ținută deoparte pentru test |
| **Precision / Recall / F1** | Cât de des are dreptate când zice „cocoșat” / câte cazuri de „cocoșat” prinde / un echilibru între ele |
| **Matricea de confuzie** | Un tabel care arată ce clase sunt confundate între ele |
| **Random Forest** | Mulți arbori de decizie, fiecare antrenat puțin diferit, care votează |
| **Regresie logistică** | Un model simplu care trasează granițe drepte între clase |
| **Normalizare** | Aducerea valorilor la o scară comparabilă |
| **Augmentare** | Generarea de date noi prin modificarea celor existente (rotiri, întuneric, zgomot) |
| **Model pre-antrenat** | Un model antrenat deja de altcineva (MediaPipe), pe care îl folosești ca piesă |

---

## 🎤 Întrebări de interviu

Dacă poți răspunde la toate, cunoști proiectul suficient de bine:

**De bază**
1. Explică proiectul în 2 minute.
2. Ce face MediaPipe și ce face modelul tău?
3. Cum ai strâns și etichetat datele? Câte exemple ai?

**De ce?**
4. De ce ai împărțit distanțele la lățimea umerilor?
5. De ce F1 macro și nu doar acuratețe?
6. De ce Random Forest? Ce avantaj are față de regresia logistică aici?
7. De ce mediana în timp pentru caracteristici, dar media pentru probabilități?

**Gândire critică**
8. Rezultatul cu împărțirea aleatorie era 0.99. De ce nu e corect?
9. Unde greșește modelul? Ce date ai mai strânge?
10. Ce e mai grav aici: o alertă falsă sau o postură greșită care nu e detectată?
11. De ce CLAHE pe canalul L și nu pe R, G, B?
12. Ce s-ar întâmpla cu camera pusă mult mai sus sau pentru un om de 2 m?
13. Ce ai face diferit în versiunea 2? (de ex.: CNN pe imagine, mai mulți oameni, calibrare personală)

**Teorie legată de proiect**
14. Ce e overfitting-ul? L-ai întâlnit?
15. Ce e o convoluție? Care e legătura dintre Sobel și un CNN?

---

*Proiect realizat în cadrul pregătirii pentru un post de Junior ML Engineer.
Tehnici de preprocesare preluate din laboratoarele de Prelucrarea Imaginilor (An 3).*
