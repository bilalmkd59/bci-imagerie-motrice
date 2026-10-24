"""Phase 3 : géométrie riemannienne (espace tangent + régression logistique, MDM).

Usage (depuis la racine du projet, après la phase 2) :
    .venv\\Scripts\\python.exe -m src.phase3_riemann

Les trois méthodes (CSP+LDA, TS+LR, MDM) sont évaluées dans le même lancement, avec la même
boucle et les mêmes plis. La comparaison est donc appariée : même sujet, mêmes essais, mêmes plis.
Les scores CSP+LDA sont comparés à ceux de la phase 2 (contrôle de reproductibilité).

Produit :
    results/phase3_scores.csv, phase3_predictions.csv     scores et prédictions des 3 méthodes
    results/phase3_resume.csv                              moyenne ± écart-type, 3 méthodes
    results/phase3_wilcoxon.csv                            tests appariés entre méthodes
    reports/figures/phase3_*.png
"""

import os

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")  # aussi pour les processus parallèles

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from pyriemann.geometry.geodesic import geodesic_euclid, geodesic_riemann
from scipy.stats import wilcoxon

from src.chargement import fixer_graine
from src.experiences import SUJETS, TACHES, evaluer_tous, resumer
from src.figures import ENCRE_DISCRETE, ENCRE_SECONDAIRE, RACINE, appliquer_style, sauvegarder
from src.pipelines import csp_lda, mdm, ts_lr

DOSSIER_RESULTATS = RACINE / "results"
ORDRE_METHODES = ["CSP+LDA", "TS+LR", "MDM"]
COULEURS_METHODES = {"CSP+LDA": "#2a78d6", "TS+LR": "#eb6834", "MDM": "#1baf7a"}
NOMS_TACHES = {"binaire": "Main gauche contre main droite", "4classes": "4 classes"}
NOMS_EVALUATIONS = {"intra": "intra-session", "inter": "inter-session"}
COMPARAISONS = [("TS+LR", "CSP+LDA"), ("MDM", "CSP+LDA"), ("TS+LR", "MDM")]


def correction_holm(p_valeurs: np.ndarray) -> np.ndarray:
    """Correction de Holm-Bonferroni pour comparaisons multiples (contrôle du risque global)."""
    p_valeurs = np.asarray(p_valeurs, dtype=float)
    m = len(p_valeurs)
    ordre = np.argsort(p_valeurs)
    corrigees = np.empty(m)
    maximum_courant = 0.0
    for rang, indice in enumerate(ordre):
        maximum_courant = max(maximum_courant, (m - rang) * p_valeurs[indice])
        corrigees[indice] = min(1.0, maximum_courant)
    return corrigees


def tests_apparies(par_sujet: pd.DataFrame) -> pd.DataFrame:
    """Test de Wilcoxon apparié (rangs signés, bilatéral) sur les kappas des 9 sujets."""
    lignes = []
    for tache in TACHES:
        for evaluation in ["intra", "inter"]:
            sous = par_sujet[(par_sujet["tache"] == tache) & (par_sujet["evaluation"] == evaluation)]
            kappa = sous.pivot(index="sujet", columns="methode", values="kappa")
            for a, b in COMPARAISONS:
                difference = kappa[a] - kappa[b]
                test = wilcoxon(kappa[a], kappa[b])
                lignes.append({
                    "tache": tache, "evaluation": evaluation, "methode_a": a, "methode_b": b,
                    "kappa_moyen_a": kappa[a].mean(), "kappa_moyen_b": kappa[b].mean(),
                    "difference_moyenne": difference.mean(),
                    "difference_mediane": difference.median(),
                    "sujets_a_meilleur": int((difference > 0).sum()),
                    "sujets_egalite": int((difference == 0).sum()),
                    "statistique_W": test.statistic, "p_valeur": test.pvalue,
                })
    tests = pd.DataFrame(lignes)
    tests["p_valeur_holm"] = correction_holm(tests["p_valeur"])
    return tests


def figure_comparaison(par_sujet: pd.DataFrame, tests: pd.DataFrame, nom: str) -> None:
    """Kappa par sujet pour les 3 méthodes : un trait par sujet relie ses 3 scores."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharey=True, constrained_layout=True)
    positions = np.arange(len(ORDRE_METHODES))
    for i, tache in enumerate(TACHES):
        for j, evaluation in enumerate(["intra", "inter"]):
            ax = axes[i, j]
            sous = par_sujet[(par_sujet["tache"] == tache) & (par_sujet["evaluation"] == evaluation)]
            kappa = sous.pivot(index="sujet", columns="methode", values="kappa")[ORDRE_METHODES]
            for sujet, ligne in kappa.iterrows():
                ax.plot(positions, ligne.to_numpy(), color="#c3c2b7", lw=1, zorder=1)
                ax.text(positions[-1] + 0.08, ligne.iloc[-1], str(sujet), fontsize=7,
                        color=ENCRE_DISCRETE, va="center")
            for k, methode in enumerate(ORDRE_METHODES):
                ax.scatter(np.full(len(kappa), k), kappa[methode], s=22, zorder=2,
                           color=COULEURS_METHODES[methode], edgecolors="white", linewidths=0.8)
                moyenne = kappa[methode].mean()
                ax.plot([k - 0.18, k + 0.18], [moyenne, moyenne], color=ENCRE_SECONDAIRE,
                        lw=2.5, zorder=3)
                ax.text(k - 0.2, moyenne, f"{moyenne:.2f}", ha="right", va="center", fontsize=8,
                        color=ENCRE_SECONDAIRE)
            ax.axhline(0, color=ENCRE_DISCRETE, lw=0.8, ls="--")
            ax.set_xticks(positions, ORDRE_METHODES)
            ax.set_xlim(-0.6, len(ORDRE_METHODES) - 0.4)
            ax.grid(axis="y")
            t = tests[(tests["tache"] == tache) & (tests["evaluation"] == evaluation)]
            legende_tests = " ; ".join(
                f"{r.methode_a} vs {r.methode_b} : p = {r.p_valeur:.3f} (Holm {r.p_valeur_holm:.2f})"
                for r in t.itertuples() if r.methode_b == "CSP+LDA")
            ax.set_title(f"{NOMS_TACHES[tache]}, {NOMS_EVALUATIONS[evaluation]}\n"
                         f"{legende_tests}", fontsize=8)
            if j == 0:
                ax.set_ylabel("Kappa de Cohen")
    fig.suptitle("Comparaison appariée des méthodes : un trait gris par sujet, "
                 "trait foncé = moyenne des 9 sujets (Wilcoxon, n = 9)")
    sauvegarder(fig, nom)


def figure_barres(par_sujet: pd.DataFrame, nom: str) -> None:
    """Kappa par sujet et par méthode, 4 classes, inter-session (protocole de la compétition)."""
    sous = par_sujet[(par_sujet["tache"] == "4classes") & (par_sujet["evaluation"] == "inter")]
    kappa = sous.pivot(index="sujet", columns="methode", values="kappa")[ORDRE_METHODES]
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    largeur = 0.27
    for k, methode in enumerate(ORDRE_METHODES):
        ax.bar(kappa.index + (k - 1) * largeur, kappa[methode], width=largeur - 0.03,
               color=COULEURS_METHODES[methode],
               label=f"{methode} (moyenne {kappa[methode].mean():.2f})")
    ax.axhline(0, color=ENCRE_SECONDAIRE, lw=0.8)
    ax.set_xticks(SUJETS)
    ax.set_xlabel("Sujet")
    ax.set_ylabel("Kappa de Cohen")
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncols=3)
    ax.set_title("4 classes, apprentissage session 1, test session 2 : kappa par sujet et par méthode")
    sauvegarder(fig, nom)


def _ellipse(ax, matrice: np.ndarray, centre: tuple, couleur: str) -> None:
    """Trace l'ellipse {x : x' C^-1 x = 1}, image du cercle unité par C^(1/2)."""
    valeurs, vecteurs = np.linalg.eigh(matrice)
    angle = np.degrees(np.arctan2(vecteurs[1, 1], vecteurs[0, 1]))
    ax.add_patch(Ellipse(centre, 2 * np.sqrt(valeurs[1]), 2 * np.sqrt(valeurs[0]), angle=angle,
                         facecolor=couleur, alpha=0.35, edgecolor=couleur, lw=1.5))


def figure_intuition(nom: str) -> None:
    """Interpolation entre deux covariances 2 x 2 : chemin euclidien contre géodésique riemannienne.

    A et B ont le même déterminant (même « volume » de signal), seule leur orientation diffère.
    Le chemin euclidien (moyenne terme à terme) gonfle le déterminant au milieu : c'est
    l'effet de gonflement. La géodésique riemannienne conserve le déterminant.
    """
    A = np.diag([4.0, 0.25])
    rotation = np.array([[np.cos(np.pi / 3), -np.sin(np.pi / 3)],
                         [np.sin(np.pi / 3), np.cos(np.pi / 3)]])
    B = rotation @ A @ rotation.T
    etapes = np.linspace(0, 1, 5)
    chemins = {"Chemin euclidien (moyenne terme à terme)": (geodesic_euclid, "#2a78d6"),
               "Géodésique riemannienne": (geodesic_riemann, "#eb6834")}

    fig = plt.figure(figsize=(12, 4.6), constrained_layout=True)
    grille = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
    for ligne, (titre, (geodesique, couleur)) in enumerate(chemins.items()):
        ax = fig.add_subplot(grille[ligne, 0])
        for k, t in enumerate(etapes):
            C = geodesique(A, B, t)
            _ellipse(ax, C, (5 * k, 0), couleur)
            ax.text(5 * k, -2.6, f"t = {t:.2f}\ndét = {np.linalg.det(C):.2f}", ha="center",
                    fontsize=8, color=ENCRE_SECONDAIRE)
        ax.set_xlim(-2.8, 5 * (len(etapes) - 1) + 2.8)
        ax.set_ylim(-3.4, 2.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(titre, fontsize=10, loc="left")

    ax = fig.add_subplot(grille[:, 1])
    t_fin = np.linspace(0, 1, 101)
    for titre, (geodesique, couleur) in chemins.items():
        ax.plot(t_fin, [np.linalg.det(geodesique(A, B, t)) for t in t_fin], color=couleur,
                label=titre.split(" (")[0])
    ax.set_xlabel("Position t le long du chemin, de A (t = 0) à B (t = 1)")
    ax.set_ylabel("Déterminant (volume de l'ellipse)")
    ax.set_ylim(0, None)
    ax.grid(axis="y")
    ax.legend(loc="lower center")
    ax.set_title("Effet de gonflement", fontsize=10)
    fig.suptitle("Pourquoi une géométrie riemannienne : interpoler entre deux matrices de covariance "
                 "de même volume mais d'orientations différentes")
    sauvegarder(fig, nom)


def main() -> None:
    fixer_graine()
    mne.set_log_level("ERROR")
    appliquer_style()

    # Les trois méthodes dans le même lancement : mêmes essais, mêmes plis.
    # Pas de permutations ici : le niveau du hasard dépend du protocole et a été mesuré en phase 2 ;
    # la comparaison entre méthodes repose sur le test de Wilcoxon apparié.
    scores, predictions = evaluer_tous({"CSP+LDA": csp_lda, "TS+LR": ts_lr, "MDM": mdm},
                                       avec_hasard=False)
    scores.round(4).to_csv(DOSSIER_RESULTATS / "phase3_scores.csv", index=False)
    predictions.to_csv(DOSSIER_RESULTATS / "phase3_predictions.csv", index=False)

    # Contrôle de reproductibilité : CSP + LDA doit redonner exactement les scores de la phase 2
    cles = ["sujet", "tache", "evaluation", "session"]
    phase2 = pd.read_csv(DOSSIER_RESULTATS / "phase2_scores.csv").set_index(cles)
    refait = scores[scores["methode"] == "CSP+LDA"].set_index(cles).round(4)
    ecart = (refait[["accuracy", "kappa"]] - phase2.loc[refait.index, ["accuracy", "kappa"]]).abs()
    print(f"\nContrôle : écart maximal avec les scores CSP+LDA de la phase 2 = {ecart.max().max():.4f}")
    if ecart.max().max() > 1e-4:
        raise RuntimeError("Les scores CSP+LDA ne sont pas reproduits : arrêt.")

    par_sujet, resume = resumer(scores)
    par_sujet.round(4).to_csv(DOSSIER_RESULTATS / "phase3_scores_par_sujet.csv", index=False)
    resume.round(4).to_csv(DOSSIER_RESULTATS / "phase3_resume.csv", index=False)
    tests = tests_apparies(par_sujet)
    tests.round(4).to_csv(DOSSIER_RESULTATS / "phase3_wilcoxon.csv", index=False)

    figure_comparaison(par_sujet, tests, "phase3_comparaison_methodes.png")
    figure_barres(par_sujet, "phase3_kappa_4classes_inter.png")
    figure_intuition("phase3_intuition_variete.png")

    print("\nRésumé sur les 9 sujets :")
    print(resume.round(3).to_string(index=False))
    print("\nTests de Wilcoxon appariés (kappa, 9 sujets) :")
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
