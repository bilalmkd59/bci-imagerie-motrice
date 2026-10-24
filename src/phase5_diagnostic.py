"""Phase 5, diagnostic : stabilité d'EEGNet selon la graine et écart apprentissage / test.

Diagnostic seulement : ces chiffres ne servent à choisir aucun réglage (le faire au vu des
scores de test serait une fuite par sélection de modèle).

Usage : .venv\Scripts\python.exe -m src.phase5_diagnostic
Produit : results/phase5_diagnostic.csv
"""

import os

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import pandas as pd

from src.apprentissage_profond import ClassifieurEEGNet
from src.chargement import charger_epoques
from src.evaluation import scores
from src.figures import RACINE

CAS = [(7, "binaire"), (3, "binaire")]  # le sujet le plus dégradé et un sujet qui réussit
GRAINES = [42, 1, 2]


def main() -> None:
    lignes = []
    for sujet, tache in CAS:
        X, y, meta = charger_epoques(sujet, tache)
        session = meta["session"].to_numpy()
        app, test = session == "0train", session == "1test"
        for graine in GRAINES:
            modele = ClassifieurEEGNet(graine=graine).fit(X[app], y[app])
            s_app = scores(y[app], modele.predict(X[app]))
            s_test = scores(y[test], modele.predict(X[test]))
            lignes.append({"sujet": sujet, "tache": tache, "graine": graine,
                           "kappa_apprentissage": s_app["kappa"], "kappa_test": s_test["kappa"],
                           "accuracy_apprentissage": s_app["accuracy"],
                           "accuracy_test": s_test["accuracy"]})
            print(lignes[-1], flush=True)
    tableau = pd.DataFrame(lignes).round(4)
    tableau.to_csv(RACINE / "results" / "phase5_diagnostic.csv", index=False)
    print(tableau.to_string(index=False))


if __name__ == "__main__":
    main()
