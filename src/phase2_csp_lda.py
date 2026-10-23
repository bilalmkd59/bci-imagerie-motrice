"""Phase 2 : référence classique CSP + LDA, en binaire puis en 4 classes.

Usage (depuis la racine du projet) :
    .venv\\Scripts\\python.exe -m src.phase2_csp_lda

Produit :
    results/phase2_scores.csv          scores par sujet, tâche, évaluation et session
    results/phase2_resume.csv          moyenne et écart-type sur les 9 sujets
    results/phase2_predictions.csv     prédictions essai par essai (matrices de confusion)
    reports/figures/phase2_*.png
"""

import os

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")  # aussi pour les processus parallèles

from functools import partial

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.chargement import charger_epoques, fixer_graine
from src.evaluation import evaluer_inter_session, evaluer_intra_session
from src.figures import (CARTE_ERD, COULEURS_CLASSES, ENCRE_DISCRETE, ENCRE_SECONDAIRE,
                         NOMS_CLASSES, RACINE, appliquer_style, sauvegarder)
from src.pipelines import csp_lda

SUJETS = list(range(1, 10))
TACHES = ["binaire", "4classes"]
NOMS_TACHES = {"binaire": "Main gauche contre main droite", "4classes": "4 classes"}
NOMS_EVALUATIONS = {"intra": "Intra-session (validation croisée 5 plis)",
                    "inter": "Inter-session (session 1 vers session 2)"}
METHODE = "CSP+LDA"
DOSSIER_RESULTATS = RACINE / "results"


def evaluer_sujet(sujet: int, tache: str) -> tuple[list, list]:
    """Évalue CSP + LDA sur un sujet : intra-session (chaque session) puis inter-session."""
    X, y, meta = charger_epoques(sujet, tache)
    session = meta["session"].to_numpy()
    fabrique = partial(csp_lda, tache)
    lignes, predictions = [], []

    def enregistrer(evaluation, nom_session, resultat, y_vrai, y_pred):
        lignes.append({"methode": METHODE, "sujet": sujet, "tache": tache,
                       "evaluation": evaluation, "session": nom_session,
                       "n_essais_test": len(y_vrai), **resultat})
        predictions.append(pd.DataFrame({"methode": METHODE, "sujet": sujet, "tache": tache,
                                         "evaluation": evaluation, "session": nom_session,
                                         "y_vrai": y_vrai, "y_pred": y_pred}))

    for nom_session in ["0train", "1test"]:
        masque = session == nom_session
        resultat, y_pred = evaluer_intra_session(fabrique, X[masque], y[masque])
        enregistrer("intra", nom_session, resultat, y[masque], y_pred)

    app, test = session == "0train", session == "1test"
    resultat, y_pred = evaluer_inter_session(fabrique, X[app], y[app], X[test], y[test])
    enregistrer("inter", "0train->1test", resultat, y[test], y_pred)
    return lignes, predictions


def resumer(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Un score par sujet (intra = moyenne des deux sessions), puis moyenne ± écart-type."""
    par_sujet = (scores.groupby(["methode", "tache", "evaluation", "sujet"], as_index=False)
                 [["accuracy", "kappa", "hasard_accuracy_p95", "p_valeur"]]
                 .agg({"accuracy": "mean", "kappa": "mean", "hasard_accuracy_p95": "mean",
                       "p_valeur": "max"}))
    par_sujet["au_dessus_du_hasard"] = par_sujet["p_valeur"] < 0.05
    resume = par_sujet.groupby(["methode", "tache", "evaluation"]).agg(
        accuracy_moyenne=("accuracy", "mean"), accuracy_ecart_type=("accuracy", "std"),
        kappa_moyen=("kappa", "mean"), kappa_ecart_type=("kappa", "std"),
        kappa_min=("kappa", "min"), kappa_max=("kappa", "max"),
        sujets_au_dessus_du_hasard=("au_dessus_du_hasard", "sum"),
    ).reset_index()
    return par_sujet, resume


def figure_scores(par_sujet: pd.DataFrame, nom: str) -> None:
    """Accuracy par sujet, avec le hasard théorique et le seuil de hasard estimé par permutation."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, constrained_layout=True)
    for i, tache in enumerate(TACHES):
        hasard_theorique = 0.5 if tache == "binaire" else 0.25
        for j, evaluation in enumerate(["intra", "inter"]):
            ax = axes[i, j]
            sous = par_sujet[(par_sujet["tache"] == tache)
                             & (par_sujet["evaluation"] == evaluation)].sort_values("sujet")
            ax.bar(sous["sujet"], sous["accuracy"], width=0.6, color="#2a78d6")
            ax.scatter(sous["sujet"], sous["hasard_accuracy_p95"], marker="_", s=260,
                       color=ENCRE_SECONDAIRE, linewidths=2, zorder=3,
                       label="95e centile du hasard (permutations)")
            ax.axhline(hasard_theorique, color=ENCRE_DISCRETE, lw=0.8, ls="--",
                       label=f"hasard théorique ({hasard_theorique:.0%})")
            for _, ligne in sous.iterrows():
                ax.text(ligne["sujet"], ligne["accuracy"] + 0.015, f"{ligne['kappa']:.2f}",
                        ha="center", fontsize=7, color=ENCRE_SECONDAIRE)
            ax.set_ylim(0, 1.08)
            ax.set_xticks(SUJETS)
            ax.grid(axis="y")
            ax.set_title(f"{NOMS_TACHES[tache]}\n{NOMS_EVALUATIONS[evaluation]}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Accuracy")
            if i == 1:
                ax.set_xlabel("Sujet")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle("CSP + LDA : accuracy par sujet (chiffre au-dessus des barres = kappa de Cohen)")
    sauvegarder(fig, nom)


def figure_confusion(predictions: pd.DataFrame, nom: str) -> None:
    """Matrices de confusion 4 classes, cumulées sur les 9 sujets et normalisées par ligne."""
    classes = list(COULEURS_CLASSES)
    etiquettes = [NOMS_CLASSES[c] for c in classes]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, evaluation in zip(axes, ["intra", "inter"]):
        sous = predictions[(predictions["tache"] == "4classes")
                           & (predictions["evaluation"] == evaluation)]
        matrice = confusion_matrix(sous["y_vrai"], sous["y_pred"], labels=classes,
                                   normalize="true")
        image = ax.imshow(matrice, cmap="Blues", vmin=0, vmax=1)
        for a in range(len(classes)):
            for b in range(len(classes)):
                ax.text(b, a, f"{matrice[a, b]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if matrice[a, b] > 0.5 else "#0b0b0b")
        ax.set_xticks(range(len(classes)), etiquettes, rotation=20)
        ax.set_yticks(range(len(classes)), etiquettes)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
        ax.set_title(NOMS_EVALUATIONS[evaluation], fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Part des essais de la classe réelle")
    fig.suptitle("CSP + LDA, 4 classes : matrices de confusion cumulées sur les 9 sujets")
    sauvegarder(fig, nom)


def figure_csp(sujet: int, nom: str) -> None:
    """Filtres et motifs CSP (binaire), appris sur toute la session 1 du sujet."""
    X, y, meta, epoques = charger_epoques(sujet, "binaire", avec_epoques=True)
    app = meta["session"].to_numpy() == "0train"
    modele = csp_lda("binaire").fit(X[app], y[app])
    csp = modele.named_steps["csp"]
    info = epoques.info
    n = csp.n_components
    # Classe pour laquelle chaque composante a la plus grande puissance (moyenne du log-variance)
    caracteristiques = csp.transform(X[app])
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.6), constrained_layout=True)
    for k in range(n):
        moyennes = {c: caracteristiques[y[app] == c, k].mean() for c in np.unique(y[app])}
        classe_forte = max(moyennes, key=moyennes.get)
        for ligne, (matrice, titre) in enumerate([(csp.patterns_, "Motif"),
                                                  (csp.filters_, "Filtre")]):
            valeurs = matrice[k]
            limite = np.abs(valeurs).max()
            mne.viz.plot_topomap(valeurs, info, axes=axes[ligne, k], cmap=CARTE_ERD,
                                 vlim=(-limite, limite), show=False, contours=0, sensors=True)
            texte = f"{titre} {k + 1}"
            if ligne == 0:
                texte += f"\npuissance plus forte :\n{NOMS_CLASSES[classe_forte].lower()}"
            axes[ligne, k].set_title(texte, fontsize=8)
    fig.suptitle(f"Sujet {sujet} : composantes CSP main gauche contre main droite "
                 "(ligne du haut : motifs, ligne du bas : filtres ; unités arbitraires)",
                 fontsize=10)
    sauvegarder(fig, nom)


def main() -> None:
    fixer_graine()
    mne.set_log_level("ERROR")
    appliquer_style()

    lignes, predictions = [], []
    for tache in TACHES:
        for sujet in SUJETS:
            l, p = evaluer_sujet(sujet, tache)
            lignes += l
            predictions += p
            resume_sujet = ", ".join(f"{x['evaluation']} {x['session']} : acc {x['accuracy']:.3f} "
                                     f"kappa {x['kappa']:.3f}" for x in l)
            print(f"{tache} sujet {sujet} | {resume_sujet}", flush=True)

    scores = pd.DataFrame(lignes)
    predictions = pd.concat(predictions, ignore_index=True)
    scores.round(4).to_csv(DOSSIER_RESULTATS / "phase2_scores.csv", index=False)
    predictions.to_csv(DOSSIER_RESULTATS / "phase2_predictions.csv", index=False)
    par_sujet, resume = resumer(scores)
    par_sujet.round(4).to_csv(DOSSIER_RESULTATS / "phase2_scores_par_sujet.csv", index=False)
    resume.round(4).to_csv(DOSSIER_RESULTATS / "phase2_resume.csv", index=False)

    figure_scores(par_sujet, "phase2_scores_par_sujet.png")
    figure_confusion(predictions, "phase2_confusion_4classes.png")
    # Sujets illustrés : le sujet 1, puis le meilleur et le moins bon selon la validation
    # croisée sur la session 1 (choix qui ne regarde pas la session de test).
    intra_binaire = scores[(scores["tache"] == "binaire") & (scores["evaluation"] == "intra")
                           & (scores["session"] == "0train")].set_index("sujet")["accuracy"]
    for sujet in dict.fromkeys([1, intra_binaire.idxmax(), intra_binaire.idxmin()]):
        figure_csp(int(sujet), f"phase2_csp_sujet{int(sujet):02d}.png")

    print("\nRésumé sur les 9 sujets :")
    print(resume.round(3).to_string(index=False))
    print("\nScores par sujet :")
    print(par_sujet.pivot_table(index="sujet", columns=["tache", "evaluation"],
                                values="kappa").round(3).to_string())


if __name__ == "__main__":
    main()
