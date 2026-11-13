import glob
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import NUME_CARACTERISTICI

date = pd.concat([pd.read_csv(f) for f in glob.glob("data/*.csv")], ignore_index=True)
X = date[NUME_CARACTERISTICI]
y = date["eticheta"]
grupuri = date["persoana"] if date["persoana"].nunique() > 1 else date["sesiune"]

# Cel mai bun model pe grupuri noi în Ziua 5
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# Fiecare rând e prezis de un model care NU i-a văzut grupul la antrenare
predictii = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=grupuri)
print(classification_report(y, predictii))

# Matricea de confuzie: ce postură e confundată cu care
os.makedirs("results", exist_ok=True)
ConfusionMatrixDisplay.from_predictions(y, predictii, cmap="Blues")
plt.title("Matricea de confuzie (testat pe grupuri noi)")
plt.savefig("results/matrice_confuzie.png", dpi=150)
plt.close()
print("Salvat: results/matrice_confuzie.png")

# Pe ce grup (persoană sau sesiune) greșește cel mai des?
date["corect"] = predictii == y
print("\nAcuratete pe fiecare grup:")
print(date.groupby(grupuri)["corect"].mean().round(2))

# Antrenăm modelul final pe TOATE datele
model.fit(X, y)

# Cât scade scorul dacă amestecăm o caracteristică? Cu cât scade mai mult, cu atât contează mai mult
rezultat = permutation_importance(model, X, y, n_repeats=5, random_state=0, scoring="f1_macro")
importanta = pd.Series(rezultat.importances_mean, index=NUME_CARACTERISTICI).sort_values()
print("\nImportanta caracteristicilor:")
print(importanta.round(3))

importanta.plot.barh(title="Importanta caracteristicilor")
plt.tight_layout()
plt.savefig("results/importanta.png", dpi=150)
plt.close()
print("Salvat: results/importanta.png")

# Salvăm modelul final ca să-l folosim în aplicația live (Ziua 7)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
print("\nModel salvat in models/model.joblib")
