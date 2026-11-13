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

X = date[NUME_CARACTERISTICI]   # intrările: cele 6 numere
y = date["eticheta"]            # răspunsul corect: drept / cocosat / aplecat

# Testăm pe persoane noi; dacă avem o singură persoană, măcar pe sesiuni noi
if date["persoana"].nunique() > 1:
    grupuri = date["persoana"]
else:
    grupuri = date["sesiune"]

print("X:", X.shape)
print("y:", y.shape)
print("Grupuri:", grupuri.unique())


modele = {
    "Baseline (clasa cea mai frecventa)": DummyClassifier(strategy="most_frequent"),
    "Regresie logistica": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
}

print("Modele pregatite:", list(modele.keys()))


print(f"\n{'Model':40s} {'Aleator (GRESIT)':>18s} {'Grupuri noi (CORECT)':>22s}")
for nume, model in modele.items():
    # GREȘIT: amestecăm rândurile -> cadre aproape identice ajung și în antrenare, și în test
    gresit = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0),
                             scoring="f1_macro")
    # CORECT: testăm pe o sesiune/persoană pe care modelul n-a văzut-o deloc
    corect = cross_val_score(model, X, y, cv=LeaveOneGroupOut(), groups=grupuri,
                             scoring="f1_macro")
    print(f"{nume:40s} {gresit.mean():18.2f} {corect.mean():22.2f}")


