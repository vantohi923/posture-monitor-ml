import glob

import matplotlib.pyplot as plt
import pandas as pd

from features import NUME_CARACTERISTICI

fisiere = glob.glob("data/*.csv")
print("Fisiere gasite:", fisiere)

date = pd.concat([pd.read_csv(f) for f in fisiere], ignore_index=True)
print("Total randuri:", len(date))
print(date.head())


print("\nRanduri pe postura:")
print(date["eticheta"].value_counts())

print("\nRanduri pe persoana:")
print(date["persoana"].value_counts())

print("\nRanduri pe sesiune:")
print(date["sesiune"].value_counts())


print("\nMedia fiecarei caracteristici pe postura:")
print(date.groupby("eticheta")[NUME_CARACTERISTICI].mean().round(2))


# Pentru fiecare caracteristică: cum diferă valorile între posturi?
date.boxplot(column=NUME_CARACTERISTICI, by="eticheta", layout=(2, 3), figsize=(12, 7))
plt.suptitle("Caracteristicile pe fiecare postura")
plt.tight_layout()
plt.show()



