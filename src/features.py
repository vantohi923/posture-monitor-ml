"""Transformă cele 33 de puncte MediaPipe în câteva numere care descriu postura."""
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
