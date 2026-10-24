"""Phase 4 : généralisation inter-sujets (leave-one-subject-out), sans et avec recentrage.

Usage (depuis la racine du projet, après la phase 3) :
    .venv\\Scripts\\python.exe -m src.phase4_inter_sujets

Protocole, pour chaque sujet testé s :
    apprentissage : les 2 sessions des 8 autres sujets (aucun essai de s) ;
    test          : la session 2 de s, c'est-à-dire exactement les essais de test de
                    l'évaluation inter-session des phases 2 et 3 (comparaison appariée avec
                    un modèle propre au sujet, appris sur sa session 1).
Recentrage (src/alignement.py), référence du sujet testé calculée sans étiquette sur :
    "calibration"  : sa session 1 (aucun essai de test utilisé) ;
    "transductif"  : ses essais de test eux-mêmes (variante courante, plus optimiste).

Produit :
    results/phase4_scores.csv, phase4_resume.csv, phase4_wilcoxon.csv
    reports/figures/phase4_*.png
"""

import os

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.geometry.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA

from src.alignement import (ClassifieurRecentre, mdm_sur_covariances, recentrer,
                            ts_lr_sur_covariances)
from src.chargement import charger_epoques, fixer_graine
from src.evaluation import scores
from src.experiences import SUJETS, TACHES
from src.figures import (COULEURS_CLASSES, ENCRE_DISCRETE, ENCRE_SECONDAIRE, NOMS_CLASSES, RACINE,
                         appliquer_style, sauvegarder)
from src.phase3_riemann import correction_holm
from src.pipelines import ESTIMATEUR_COVARIANCE, csp_lda

DOSSIER_RESULTATS = RACINE / "results"
NOMS_TACHES = {"binaire": "Main gauche contre main droite", "4classes": "4 classes"}

# Approches comparées : (méthode, alignement). « propre au sujet » = inter-session de la phase 3.
APPROCHES = [
    ("TS+LR", "propre au sujet"), ("TS+LR", "aucun"), ("TS+LR", "calibration"),
    ("TS+LR", "transductif"),
    ("MDM", "propre au sujet"), ("MDM", "aucun"), ("MDM", "calibration"), ("MDM", "transductif"),
    ("CSP+LDA", "propre au sujet"), ("CSP+LDA", "aucun"),
]
NOMS_ALIGNEMENTS = {
    "propre au sujet": "Modèle propre au sujet\n(sa session 1)",
    "aucun": "Autres sujets,\nsans recentrage",
    "calibration": "Autres sujets, recentrage\n(référence : session 1)",
    "transductif": "Autres sujets, recentrage\n(référence : essais de test)",
}
COULEURS_ALIGNEMENTS = {"propre au sujet": "#2a78d6", "aucun": "#eb6834",
                        "calibration": "#1baf7a", "transductif": "#eda100"}
COMPARAISONS = [
    # (méthode, alignement A, alignement B) : test de A contre B
    ("TS+LR", "aucun", "propre au sujet"),
    ("TS+LR", "calibration", "aucun"),
    ("TS+LR", "transductif", "aucun"),
    ("TS+LR", "calibration", "propre au sujet"),
    ("MDM", "calibration", "aucun"),
    ("MDM", "transductif", "aucun"),
]


def charger_tous(tache: str) -> dict:
    """Charge les essais des 9 sujets et estime leurs covariances (une fois pour toutes).

    L'estimation OAS est faite essai par essai, indépendamment des autres essais et sans
    étiquette : la précalculer ne crée aucune fuite.
    """
    donnees = {}
    for sujet in SUJETS:
        X, y, meta = charger_epoques(sujet, tache)
        donnees[sujet] = {
            "X": X, "y": y, "session": meta["session"].to_numpy(),
            "C": Covariances(estimator=ESTIMATEUR_COVARIANCE).transform(X),
        }
    return donnees


def evaluer_loso(donnees: dict, tache: str) -> list:
    """Leave-one-subject-out : 9 plis, un par sujet testé."""
    lignes = []
    for sujet_test in SUJETS:
        autres = [s for s in SUJETS if s != sujet_test]
        X_app = np.concatenate([donnees[s]["X"] for s in autres])
        C_app = np.concatenate([donnees[s]["C"] for s in autres])
        y_app = np.concatenate([donnees[s]["y"] for s in autres])
        groupes = np.concatenate([[f"{s}-{session}" for session in donnees[s]["session"]]
                                  for s in autres])

        d = donnees[sujet_test]
        test, calibration = d["session"] == "1test", d["session"] == "0train"
        X_test, C_test, y_test = d["X"][test], d["C"][test], d["y"][test]
        C_calibration = d["C"][calibration]  # étiquettes jamais utilisées

        predictions = {
            ("CSP+LDA", "aucun"): csp_lda(tache).fit(X_app, y_app).predict(X_test),
            ("TS+LR", "aucun"): ts_lr_sur_covariances().fit(C_app, y_app).predict(C_test),
            ("MDM", "aucun"): mdm_sur_covariances().fit(C_app, y_app).predict(C_test),
        }
        for methode, fabrique in [("TS+LR", ts_lr_sur_covariances), ("MDM", mdm_sur_covariances)]:
            modele = ClassifieurRecentre(fabrique).fit(C_app, y_app, groupes)
            predictions[(methode, "calibration")] = modele.predict(C_test, C_calibration)
            predictions[(methode, "transductif")] = modele.predict(C_test, C_test)

        for (methode, alignement), y_pred in predictions.items():
            lignes.append({"methode": methode, "alignement": alignement, "sujet": sujet_test,
                           "tache": tache, "n_essais_apprentissage": len(y_app),
                           "n_essais_test": len(y_test), **scores(y_test, y_pred)})
        resume = ", ".join(f"{m} {a} {scores(y_test, p)['kappa']:.3f}"
                           for (m, a), p in predictions.items())
        print(f"{tache} sujet {sujet_test} (kappa) | {resume}", flush=True)
    return lignes


def tests_apparies(kappa: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon apparié sur les 9 sujets pour chaque comparaison, correction de Holm globale."""
    lignes = []
    for tache in TACHES:
        for methode, a, b in COMPARAISONS:
            ka = kappa.loc[(tache, methode, a)]
            kb = kappa.loc[(tache, methode, b)]
            difference = ka - kb
            test = wilcoxon(ka, kb)
            lignes.append({"tache": tache, "methode": methode, "alignement_a": a,
                           "alignement_b": b, "kappa_moyen_a": ka.mean(), "kappa_moyen_b": kb.mean(),
                           "difference_moyenne": difference.mean(),
                           "sujets_a_meilleur": int((difference > 0).sum()),
                           "statistique_W": test.statistic, "p_valeur": test.pvalue})
    tests = pd.DataFrame(lignes)
    tests["p_valeur_holm"] = correction_holm(tests["p_valeur"])
    return tests


def figure_par_sujet(tableau: pd.DataFrame, nom: str) -> None:
    """Kappa par sujet (TS+LR) pour les 4 approches, binaire et 4 classes."""
    alignements = list(NOMS_ALIGNEMENTS)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True)
    largeur = 0.2
    for ax, tache in zip(axes, TACHES):
        sous = tableau[(tableau["tache"] == tache) & (tableau["methode"] == "TS+LR")]
        for k, alignement in enumerate(alignements):
            valeurs = sous[sous["alignement"] == alignement].sort_values("sujet")
            ax.bar(valeurs["sujet"] + (k - 1.5) * largeur, valeurs["kappa"], width=largeur - 0.03,
                   color=COULEURS_ALIGNEMENTS[alignement],
                   label=f"{NOMS_ALIGNEMENTS[alignement].replace(chr(10), ' ')} "
                         f"(moyenne {valeurs['kappa'].mean():.2f})")
        ax.axhline(0, color=ENCRE_SECONDAIRE, lw=0.8)
        ax.set_ylim(min(-0.1, sous["kappa"].min() - 0.05), 1.0)
        ax.set_ylabel("Kappa de Cohen")
        ax.set_title(f"{NOMS_TACHES[tache]}, TS+LR, test sur la session 2 du sujet", fontsize=10)
        ax.grid(axis="y")
        ax.legend(loc="upper left", fontsize=7, ncols=2)
    axes[-1].set_xticks(SUJETS)
    axes[-1].set_xlabel("Sujet testé")
    fig.suptitle("Généralisation à un sujet jamais vu (leave-one-subject-out)")
    sauvegarder(fig, nom)


def figure_resume(tableau: pd.DataFrame, nom: str) -> None:
    """Kappa moyen ± écart-type par méthode et par approche."""
    alignements = list(NOMS_ALIGNEMENTS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    for ax, tache in zip(axes, TACHES):
        sous = tableau[tableau["tache"] == tache]
        for k, methode in enumerate(["TS+LR", "MDM", "CSP+LDA"]):
            for j, alignement in enumerate(alignements):
                valeurs = sous[(sous["methode"] == methode) & (sous["alignement"] == alignement)]
                if valeurs.empty:
                    continue
                x = j + (k - 1) * 0.22
                moyenne, ecart = valeurs["kappa"].mean(), valeurs["kappa"].std()
                ax.errorbar(x, moyenne, yerr=ecart, fmt="o", ms=7, capsize=3, lw=1.2,
                            color=["#2a78d6", "#1baf7a", "#52514e"][k],
                            label=methode if j == 0 else None)
                ax.text(x + 0.05, moyenne, f"{moyenne:.2f}", fontsize=7, va="center",
                        color=ENCRE_SECONDAIRE)
        ax.axhline(0, color=ENCRE_DISCRETE, lw=0.8, ls="--")
        ax.set_xticks(range(len(alignements)), [NOMS_ALIGNEMENTS[a] for a in alignements],
                      fontsize=7)
        ax.set_title(NOMS_TACHES[tache], fontsize=10)
        ax.grid(axis="y")
    axes[0].set_ylabel("Kappa moyen ± écart-type (9 sujets)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Perte quand le sujet n'a jamais été vu, et gain apporté par le recentrage")
    sauvegarder(fig, nom)


def figure_espace_tangent(donnees: dict, nom: str) -> None:
    """Essais des 9 sujets (session 1) dans l'espace tangent, projetés en 2D, avant et après recentrage.

    Figure descriptive (aucune évaluation) : chaque point est un essai, coloré par classe ;
    les numéros marquent le centre de chaque sujet.
    """
    C = np.concatenate([donnees[s]["C"][donnees[s]["session"] == "0train"] for s in SUJETS])
    y = np.concatenate([donnees[s]["y"][donnees[s]["session"] == "0train"] for s in SUJETS])
    sujets = np.concatenate([np.full((donnees[s]["session"] == "0train").sum(), s) for s in SUJETS])
    C_rec = np.empty_like(C)
    for s in SUJETS:
        C_rec[sujets == s] = recentrer(C[sujets == s], mean_riemann(C[sujets == s]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for ax, (titre, covariances) in zip(axes, [("Sans recentrage", C),
                                               ("Après recentrage de chaque sujet", C_rec)]):
        vecteurs = TangentSpace(metric="riemann").fit_transform(covariances)
        plan = PCA(n_components=2).fit(vecteurs)
        points = plan.transform(vecteurs)
        for classe, couleur in COULEURS_CLASSES.items():
            masque = y == classe
            ax.scatter(points[masque, 0], points[masque, 1], s=5, alpha=0.35, color=couleur,
                       label=NOMS_CLASSES[classe], linewidths=0)
        centres = np.array([points[sujets == s].mean(axis=0) for s in SUJETS])
        if np.ptp(centres, axis=0).max() < 1e-6:
            # Après recentrage, les 9 centres coïncident à l'origine : on le dit au lieu d'empiler 9 numéros
            ax.scatter(*centres[0], s=60, color="white", edgecolors="#0b0b0b", zorder=3)
            ax.annotate("les centres des 9 sujets\nsont confondus ici", centres[0], xytext=(30, 40),
                        textcoords="offset points", fontsize=8, color="#0b0b0b",
                        arrowprops=dict(arrowstyle="->", color=ENCRE_SECONDAIRE, lw=0.8))
        else:
            for s, centre in zip(SUJETS, centres):
                ax.text(*centre, str(s), fontsize=10, fontweight="bold", ha="center", va="center",
                        color="#0b0b0b", bbox=dict(boxstyle="circle,pad=0.2", fc="white",
                                                   ec="#c3c2b7", lw=0.8))
        variance = plan.explained_variance_ratio_ * 100
        ax.set_xlabel(f"Axe principal 1 ({variance[0]:.0f} % de la variance)")
        ax.set_ylabel(f"Axe principal 2 ({variance[1]:.0f} % de la variance)")
        ax.set_title(titre, fontsize=10)
    axes[0].legend(loc="best", markerscale=3)
    fig.suptitle("4 classes, session 1 des 9 sujets dans l'espace tangent (ACP) : "
                 "un point = un essai, un numéro = le centre d'un sujet")
    sauvegarder(fig, nom)


def main() -> None:
    fixer_graine()
    mne.set_log_level("ERROR")
    appliquer_style()

    lignes = []
    donnees_4classes = None
    for tache in TACHES:
        donnees = charger_tous(tache)
        lignes += evaluer_loso(donnees, tache)
        if tache == "4classes":
            donnees_4classes = donnees
    loso = pd.DataFrame(lignes)
    loso.round(4).to_csv(DOSSIER_RESULTATS / "phase4_scores.csv", index=False)

    # Référence « modèle propre au sujet » : évaluation inter-session de la phase 3
    # (appris sur la session 1 du sujet, testé sur les mêmes essais de sa session 2)
    phase3 = pd.read_csv(DOSSIER_RESULTATS / "phase3_scores.csv")
    propre = phase3[phase3["evaluation"] == "inter"].assign(alignement="propre au sujet")
    tableau = pd.concat([loso, propre[["methode", "alignement", "sujet", "tache", "accuracy",
                                       "kappa"]]], ignore_index=True)

    resume = tableau.groupby(["tache", "methode", "alignement"]).agg(
        accuracy_moyenne=("accuracy", "mean"), accuracy_ecart_type=("accuracy", "std"),
        kappa_moyen=("kappa", "mean"), kappa_ecart_type=("kappa", "std"),
        kappa_min=("kappa", "min"), kappa_max=("kappa", "max")).reset_index()
    resume.round(4).to_csv(DOSSIER_RESULTATS / "phase4_resume.csv", index=False)

    kappa = tableau.set_index(["tache", "methode", "alignement", "sujet"])["kappa"].sort_index()
    tests = tests_apparies(kappa)
    tests.round(4).to_csv(DOSSIER_RESULTATS / "phase4_wilcoxon.csv", index=False)

    # Part de la perte inter-sujets récupérée par le recentrage (TS+LR, moyennes des 9 sujets)
    recuperation = []
    for tache in TACHES:
        k = {a: kappa.loc[(tache, "TS+LR", a)].mean() for a in NOMS_ALIGNEMENTS}
        perte = k["propre au sujet"] - k["aucun"]
        for alignement in ["calibration", "transductif"]:
            recuperation.append({"tache": tache, "alignement": alignement,
                                 "perte_kappa_sans_recentrage": perte,
                                 "gain_kappa_recentrage": k[alignement] - k["aucun"],
                                 "part_de_la_perte_recuperee": (k[alignement] - k["aucun"]) / perte})
    recuperation = pd.DataFrame(recuperation)
    recuperation.round(4).to_csv(DOSSIER_RESULTATS / "phase4_recuperation.csv", index=False)

    figure_par_sujet(tableau, "phase4_kappa_par_sujet.png")
    figure_resume(tableau, "phase4_resume_methodes.png")
    figure_espace_tangent(donnees_4classes, "phase4_espace_tangent.png")

    print("\nRésumé (kappa, 9 sujets) :")
    print(resume.round(3).to_string(index=False))
    print("\nTests de Wilcoxon appariés :")
    print(tests.round(4).to_string(index=False))
    print("\nPart de la perte récupérée (TS+LR) :")
    print(recuperation.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
