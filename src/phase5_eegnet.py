"""Phase 5 : EEGNet comparé aux méthodes précédentes, sur les mêmes évaluations.

Usage (depuis la racine du projet) :
    # en local sur CPU : protocole de la compétition (environ 1 h)
    .venv\\Scripts\\python.exe -m src.phase5_eegnet --protocoles inter
    # sur Google Colab avec GPU (voir notebooks/05_eegnet_colab.ipynb)
    python -m src.phase5_eegnet --protocoles intra inter loso --appareil cuda
    # comparaison seule, à partir des CSV présents dans results/
    .venv\\Scripts\\python.exe -m src.phase5_eegnet --comparer-seulement

Chaque protocole écrit results/phase5_scores_<protocole>.csv. La comparaison relit les scores
des phases 3 (intra, inter) et 4 (leave-one-subject-out) : mêmes essais de test, mêmes plis.
"""

import os

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import argparse
from functools import partial

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from src.apprentissage_profond import N_EPOQUES_INTER_SUJETS, aligner, eegnet
from src.chargement import charger_epoques, fixer_graine
from src.evaluation import scores
from src.experiences import SUJETS, TACHES, evaluer_tous
from src.figures import ENCRE_DISCRETE, ENCRE_SECONDAIRE, RACINE, appliquer_style, sauvegarder
from src.phase3_riemann import correction_holm

DOSSIER_RESULTATS = RACINE / "results"
PROTOCOLES = ["intra", "inter", "loso"]
NOMS_PROTOCOLES = {"intra": "intra-session", "inter": "inter-session",
                   "loso": "sujet jamais vu"}
NOMS_TACHES = {"binaire": "Main gauche contre main droite", "4classes": "4 classes"}
COULEURS_METHODES = {"CSP+LDA": "#2a78d6", "TS+LR": "#eb6834", "MDM": "#1baf7a",
                     "EEGNet": "#e87ba4"}


def evaluer_loso(tache: str, appareil: str) -> list:
    """Leave-one-subject-out avec EEGNet, sans et avec alignement euclidien (variante calibration).

    Même protocole qu'en phase 4 : apprentissage sur les 2 sessions des 8 autres sujets,
    test sur la session 2 du sujet ; la référence d'alignement du sujet testé est sa session 1,
    sans étiquette. Chaque groupe (sujet, session) d'apprentissage est aligné sur lui-même.
    """
    donnees = {}
    for sujet in SUJETS:
        X, y, meta = charger_epoques(sujet, tache)
        donnees[sujet] = {"X": X, "y": y, "session": meta["session"].to_numpy()}

    lignes = []
    for sujet_test in SUJETS:
        autres = [s for s in SUJETS if s != sujet_test]
        X_app = np.concatenate([donnees[s]["X"] for s in autres])
        y_app = np.concatenate([donnees[s]["y"] for s in autres])
        X_app_aligne = np.concatenate([
            aligner(donnees[s]["X"][donnees[s]["session"] == session],
                    donnees[s]["X"][donnees[s]["session"] == session])
            for s in autres for session in ["0train", "1test"]])
        y_app_aligne = np.concatenate([donnees[s]["y"][donnees[s]["session"] == session]
                                       for s in autres for session in ["0train", "1test"]])
        d = donnees[sujet_test]
        test, calibration = d["session"] == "1test", d["session"] == "0train"
        X_test, y_test = d["X"][test], d["y"][test]

        fabrique = partial(eegnet, tache, appareil=appareil, n_epoques=N_EPOQUES_INTER_SUJETS)
        predictions = {
            "aucun": fabrique().fit(X_app, y_app).predict(X_test),
            "calibration": fabrique().fit(X_app_aligne, y_app_aligne).predict(
                aligner(X_test, d["X"][calibration])),
        }
        for alignement, y_pred in predictions.items():
            lignes.append({"methode": "EEGNet", "alignement": alignement, "sujet": sujet_test,
                           "tache": tache, "n_essais_apprentissage": len(y_app),
                           "n_essais_test": len(y_test), **scores(y_test, y_pred)})
        print(f"{tache} sujet {sujet_test} EEGNet (kappa) | " + ", ".join(
            f"{a} {scores(y_test, p)['kappa']:.3f}" for a, p in predictions.items()), flush=True)
    return lignes


def lancer(protocoles: list, appareil: str) -> None:
    """Évalue EEGNet sur les protocoles demandés et sauvegarde un CSV par protocole."""
    for protocole in protocoles:
        if protocole in ("intra", "inter"):
            scores_eeg, predictions = evaluer_tous({"EEGNet": partial(eegnet, appareil=appareil)},
                                                   avec_hasard=False, evaluations=(protocole,))
            predictions.to_csv(DOSSIER_RESULTATS / f"phase5_predictions_{protocole}.csv",
                               index=False)
        else:
            scores_eeg = pd.DataFrame([l for tache in TACHES for l in evaluer_loso(tache, appareil)])
        scores_eeg["appareil"] = appareil
        scores_eeg.round(4).to_csv(DOSSIER_RESULTATS / f"phase5_scores_{protocole}.csv",
                                   index=False)


def tableau_comparatif() -> pd.DataFrame:
    """Kappa par sujet, méthode, tâche et protocole, pour les protocoles où EEGNet a été évalué."""
    morceaux = []
    phase3 = pd.read_csv(DOSSIER_RESULTATS / "phase3_scores.csv")
    phase3 = (phase3.groupby(["methode", "tache", "evaluation", "sujet"], as_index=False)
              [["accuracy", "kappa"]].mean().rename(columns={"evaluation": "protocole"}))
    phase4 = pd.read_csv(DOSSIER_RESULTATS / "phase4_scores.csv")
    # En leave-one-subject-out, on compare à la meilleure variante réaliste : recentrage « calibration »
    phase4 = phase4[(phase4["alignement"] == "calibration")
                    | ((phase4["methode"] == "CSP+LDA") & (phase4["alignement"] == "aucun"))]
    phase4 = phase4.assign(protocole="loso")[["methode", "tache", "protocole", "sujet",
                                               "accuracy", "kappa"]]
    for protocole in PROTOCOLES:
        chemin = DOSSIER_RESULTATS / f"phase5_scores_{protocole}.csv"
        if not chemin.exists():
            continue
        eeg = pd.read_csv(chemin)
        if protocole == "loso":
            eeg = eeg[eeg["alignement"] == "calibration"]
        eeg = (eeg.groupby(["methode", "tache", "sujet"], as_index=False)[["accuracy", "kappa"]]
               .mean().assign(protocole=protocole))
        reference = phase4 if protocole == "loso" else phase3[phase3["protocole"] == protocole]
        morceaux += [reference, eeg]
    return pd.concat(morceaux, ignore_index=True) if morceaux else pd.DataFrame()


def comparer() -> None:
    """Résumé, tests de Wilcoxon (EEGNet contre chaque méthode) et figure."""
    tableau = tableau_comparatif()
    if tableau.empty:
        print("Aucun résultat EEGNet dans results/ : rien à comparer.")
        return
    resume = tableau.groupby(["protocole", "tache", "methode"]).agg(
        accuracy_moyenne=("accuracy", "mean"), kappa_moyen=("kappa", "mean"),
        kappa_ecart_type=("kappa", "std"), kappa_min=("kappa", "min"),
        kappa_max=("kappa", "max")).reset_index()
    resume.round(4).to_csv(DOSSIER_RESULTATS / "phase5_resume.csv", index=False)

    lignes = []
    for (protocole, tache), sous in tableau.groupby(["protocole", "tache"]):
        kappa = sous.pivot(index="sujet", columns="methode", values="kappa")
        for autre in [m for m in kappa.columns if m != "EEGNet"]:
            difference = kappa["EEGNet"] - kappa[autre]
            test = wilcoxon(kappa["EEGNet"], kappa[autre])
            lignes.append({"protocole": protocole, "tache": tache, "methode_a": "EEGNet",
                           "methode_b": autre, "kappa_moyen_a": kappa["EEGNet"].mean(),
                           "kappa_moyen_b": kappa[autre].mean(),
                           "difference_moyenne": difference.mean(),
                           "sujets_a_meilleur": int((difference > 0).sum()),
                           "statistique_W": test.statistic, "p_valeur": test.pvalue})
    tests = pd.DataFrame(lignes)
    tests["p_valeur_holm"] = correction_holm(tests["p_valeur"])
    tests.round(4).to_csv(DOSSIER_RESULTATS / "phase5_wilcoxon.csv", index=False)

    # Effet de l'alignement euclidien sur EEGNet en leave-one-subject-out (si calculé)
    chemin_loso = DOSSIER_RESULTATS / "phase5_scores_loso.csv"
    if chemin_loso.exists():
        loso = pd.read_csv(chemin_loso)
        lignes_alignement = []
        for tache, sous in loso.groupby("tache"):
            kappa = sous.pivot(index="sujet", columns="alignement", values="kappa")
            test = wilcoxon(kappa["calibration"], kappa["aucun"])
            lignes_alignement.append({
                "tache": tache, "kappa_moyen_sans_alignement": kappa["aucun"].mean(),
                "kappa_moyen_avec_alignement": kappa["calibration"].mean(),
                "difference_moyenne": (kappa["calibration"] - kappa["aucun"]).mean(),
                "sujets_ameliores": int((kappa["calibration"] > kappa["aucun"]).sum()),
                "p_valeur": test.pvalue})
        alignement = pd.DataFrame(lignes_alignement)
        alignement.round(4).to_csv(DOSSIER_RESULTATS / "phase5_alignement_loso.csv", index=False)
        print("\nEEGNet, sujet jamais vu : effet de l'alignement euclidien (référence : session 1)")
        print(alignement.round(3).to_string(index=False))

    protocoles = [p for p in PROTOCOLES if p in set(tableau["protocole"])]
    fig, axes = plt.subplots(len(TACHES), len(protocoles), figsize=(5.2 * len(protocoles), 7.4),
                             sharey=True, squeeze=False, constrained_layout=True)
    for i, tache in enumerate(TACHES):
        for j, protocole in enumerate(protocoles):
            ax = axes[i, j]
            sous = tableau[(tableau["tache"] == tache) & (tableau["protocole"] == protocole)]
            kappa = sous.pivot(index="sujet", columns="methode", values="kappa")
            ordre = [m for m in COULEURS_METHODES if m in kappa.columns]
            kappa = kappa[ordre]
            positions = np.arange(len(ordre))
            for _, ligne in kappa.iterrows():
                ax.plot(positions, ligne.to_numpy(), color="#c3c2b7", lw=1, zorder=1)
            for k, methode in enumerate(ordre):
                ax.scatter(np.full(len(kappa), k), kappa[methode], s=22, zorder=2,
                           color=COULEURS_METHODES[methode], edgecolors="white", linewidths=0.8)
                moyenne = kappa[methode].mean()
                ax.plot([k - 0.18, k + 0.18], [moyenne, moyenne], color=ENCRE_SECONDAIRE, lw=2.5,
                        zorder=3)
                ax.text(k + 0.2, moyenne, f"{moyenne:.2f}", va="center", fontsize=8,
                        color=ENCRE_SECONDAIRE)
            ax.axhline(0, color=ENCRE_DISCRETE, lw=0.8, ls="--")
            ax.set_xticks(positions, ordre)
            ax.grid(axis="y")
            titre = f"{NOMS_TACHES[tache]}, {NOMS_PROTOCOLES[protocole]}"
            if protocole == "loso":
                titre += "\n(recentrage ou alignement, référence : session 1 ; CSP+LDA sans)"
            ax.set_title(titre, fontsize=9)
            if j == 0:
                ax.set_ylabel("Kappa de Cohen")
    fig.suptitle("EEGNet face aux méthodes classiques : un trait gris par sujet, "
                 "trait foncé = moyenne des 9 sujets")
    sauvegarder(fig, "phase5_comparaison_eegnet.png")

    print("\nRésumé (kappa, 9 sujets) :")
    print(resume.round(3).to_string(index=False))
    print("\nTests de Wilcoxon appariés (EEGNet contre chaque méthode) :")
    print(tests.round(4).to_string(index=False))


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--protocoles", nargs="+", choices=PROTOCOLES, default=["inter"])
    parseur.add_argument("--appareil", choices=["cpu", "cuda"],
                         default="cuda" if torch.cuda.is_available() else "cpu")
    parseur.add_argument("--comparer-seulement", action="store_true")
    arguments = parseur.parse_args()

    fixer_graine()
    mne.set_log_level("ERROR")
    appliquer_style()
    if not arguments.comparer_seulement:
        print(f"EEGNet sur {arguments.appareil}, protocoles : {arguments.protocoles}", flush=True)
        lancer(arguments.protocoles, arguments.appareil)
    comparer()


if __name__ == "__main__":
    main()
