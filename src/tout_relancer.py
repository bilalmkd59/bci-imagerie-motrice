"""Phase 6 : relance toutes les évaluations et régénère tous les tableaux, figures et notebooks.

Usage (depuis la racine du projet) :
    .venv\\Scripts\\python.exe -m src.tout_relancer                  # tout (plusieurs heures sur CPU)
    .venv\\Scripts\\python.exe -m src.tout_relancer --sans-eegnet    # sans réentraîner EEGNet
    .venv\\Scripts\\python.exe -m src.tout_relancer --etapes phase3 phase4 synthese

Chaque étape écrit son journal dans results/logs/<étape>.log ; la première erreur arrête tout.
Les durées mesurées sont enregistrées dans results/durees_execution.csv.

EEGNet en intra-session et en inter-sujets n'est pas relancé ici : trop lent sur CPU, il se calcule
avec notebooks/05_eegnet_colab.ipynb ; ses CSV, s'ils sont présents, sont repris par la comparaison.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
JOURNAUX = RACINE / "results" / "logs"
NOTEBOOKS_LOCAUX = ["01_exploration_neurophysiologique", "02_csp_lda",
                    "03_geometrie_riemannienne", "04_generalisation_inter_sujets"]

# (nom de l'étape, commande, description)
ETAPES = [
    ("environnement", ["-m", "src.verifier_environnement"], "versions et chargement du sujet 1"),
    ("phase1", ["-m", "src.phase1_exploration"], "exploration neurophysiologique (ERD)"),
    ("phase2", ["-m", "src.phase2_csp_lda"], "CSP + LDA avec 100 permutations"),
    ("phase3", ["-m", "src.phase3_riemann"], "TS+LR et MDM, tests appariés"),
    ("phase4", ["-m", "src.phase4_inter_sujets"], "leave-one-subject-out et recentrage"),
    ("phase5", ["-m", "src.phase5_eegnet", "--protocoles", "inter"], "EEGNet inter-session (CPU)"),
    ("phase5_diagnostic", ["-m", "src.phase5_diagnostic"], "EEGNet : graines et surapprentissage"),
    ("synthese", ["-m", "src.synthese"], "results/synthese.md et README"),
    ("notebooks", None, "génération et exécution des notebooks 01 à 04"),
]
ETAPES_EEGNET = {"phase5", "phase5_diagnostic"}


def commandes(nom: str, commande: list | None) -> list:
    """Liste des commandes Python à exécuter pour une étape."""
    if nom == "notebooks":
        return ([["notebooks/construire_notebooks.py", *[n[:2] for n in NOTEBOOKS_LOCAUX]]]
                + [["-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    f"notebooks/{n}.ipynb"] for n in NOTEBOOKS_LOCAUX])
    return [commande]


def lancer_etape(nom: str, commande: list | None) -> float:
    """Exécute une étape avec le Python de l'environnement courant ; renvoie sa durée en secondes."""
    environnement = os.environ | {"PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    debut = time.time()
    with open(JOURNAUX / f"{nom}.log", "w", encoding="utf-8") as journal:
        for arguments in commandes(nom, commande):
            resultat = subprocess.run([sys.executable, *arguments], cwd=RACINE, stdout=journal,
                                      stderr=subprocess.STDOUT, env=environnement)
            if resultat.returncode != 0:
                raise SystemExit(f"Échec de l'étape « {nom} » (code {resultat.returncode}) : "
                                 f"voir results/logs/{nom}.log")
    return time.time() - debut


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--etapes", nargs="+", choices=[e[0] for e in ETAPES],
                         help="n'exécuter que ces étapes (dans l'ordre du pipeline)")
    parseur.add_argument("--sans-eegnet", action="store_true",
                         help="ne pas réentraîner EEGNet ; refaire seulement sa comparaison")
    arguments = parseur.parse_args()
    JOURNAUX.mkdir(parents=True, exist_ok=True)

    a_lancer = []
    for nom, commande, description in ETAPES:
        if arguments.etapes and nom not in arguments.etapes:
            continue
        if arguments.sans_eegnet and nom in ETAPES_EEGNET:
            if nom == "phase5":
                a_lancer.append(("phase5_comparaison",
                                 ["-m", "src.phase5_eegnet", "--comparer-seulement"],
                                 "comparaison EEGNet à partir des CSV existants"))
            continue
        a_lancer.append((nom, commande, description))

    durees = []
    debut_total = time.time()
    for i, (nom, commande, description) in enumerate(a_lancer, start=1):
        print(f"[{i}/{len(a_lancer)}] {nom} : {description}...", flush=True)
        duree = lancer_etape(nom, commande)
        durees.append({"etape": nom, "description": description,
                       "duree_minutes": round(duree / 60, 1)})
        print(f"    terminé en {duree / 60:.1f} min", flush=True)
    total = (time.time() - debut_total) / 60
    print(f"Tout est terminé en {total:.1f} min.")
    if not arguments.etapes:
        pd.DataFrame(durees).to_csv(RACINE / "results" / "durees_execution.csv", index=False)


if __name__ == "__main__":
    main()
