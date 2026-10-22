"""Phase 1 : exploration neurophysiologique de l'imagerie motrice (session d'entraînement).

Usage (depuis la racine du projet) :
    .venv\\Scripts\\python.exe -m src.phase1_exploration

Produit :
    results/phase1_verification_marqueurs.csv
    results/phase1_erd_par_sujet.csv
    results/phase1_lateralisation.csv
    reports/figures/phase1_*.png
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.chargement import fixer_graine
from src.exploration import (BANDES, CANAUX_MOTEURS, FENETRE_ERD, FENETRE_REFERENCE,
                             decours_temporel, epoquer_session, erd_temps_frequence,
                             indices_lateralisation, tableau_erd, verifier_marqueurs)
from src.figures import (CARTE_ERD, COULEURS_CLASSES, ENCRE_DISCRETE, ENCRE_SECONDAIRE,
                         NOMS_CLASSES, RACINE, appliquer_style, sauvegarder)

SUJETS = list(range(1, 10))
DERIVATIONS = {"laplacien": "laplacien de surface", "ref_commune": "référence commune"}
DOSSIER_RESULTATS = RACINE / "results"
LIMITE_COULEUR = 60  # échelle commune des cartes d'ERD, en %


def _reperes_temps(ax) -> None:
    """Trace l'apparition de l'indice et la fin de l'imagerie, et grise la période de référence."""
    ax.axvspan(*FENETRE_REFERENCE, color="#e1e0d9", alpha=0.5, lw=0)
    for t in (0, 4):
        ax.axvline(t, color=ENCRE_SECONDAIRE, lw=0.8, ls="--")


def figure_temps_frequence(cartes: dict, titre: str, nom: str) -> None:
    """Cartes temps-fréquence sur C3, Cz, C4 pour main gauche et main droite."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for i, classe in enumerate(["left_hand", "right_hand"]):
        tfr = cartes[classe]
        for j, canal in enumerate(CANAUX_MOTEURS):
            ax = axes[i, j]
            image = ax.pcolormesh(tfr.times, tfr.freqs, tfr.data[tfr.ch_names.index(canal)],
                                  cmap=CARTE_ERD, vmin=-LIMITE_COULEUR, vmax=LIMITE_COULEUR,
                                  shading="nearest", rasterized=True)
            for t in (0, 4):
                ax.axvline(t, color=ENCRE_SECONDAIRE, lw=0.8, ls="--")
            for f in (8, 12, 30):
                ax.axhline(f, color="white", lw=0.6, alpha=0.8)
            ax.set_title(f"{NOMS_CLASSES[classe]}, {canal}")
            if i == 1:
                ax.set_xlabel("Temps depuis l'indice (s)")
            if j == 0:
                ax.set_ylabel("Fréquence (Hz)")
    barre = fig.colorbar(image, ax=axes, shrink=0.8)
    barre.set_label("Variation de puissance / repos (%)\nbleu = ERD, rouge = ERS")
    fig.suptitle(titre, color="#0b0b0b")
    sauvegarder(fig, nom)


def figure_topographies(cartes: dict, erd: pd.DataFrame, titre: str, nom: str) -> None:
    """Topographies de l'ERD% par classe et par bande, plus la différence main droite - main gauche."""
    info = cartes["left_hand"].info
    classes = list(COULEURS_CLASSES)
    colonnes = classes + ["difference"]
    fig, axes = plt.subplots(2, len(colonnes), figsize=(13, 5.4), constrained_layout=True)
    for i, bande in enumerate(BANDES):
        valeurs = {c: erd[(erd["bande"] == bande) & (erd["classe"] == c)]
                   .set_index("canal").loc[info.ch_names, "erd_pct"].to_numpy() for c in classes}
        valeurs["difference"] = valeurs["right_hand"] - valeurs["left_hand"]
        for j, colonne in enumerate(colonnes):
            ax = axes[i, j]
            image, _ = mne.viz.plot_topomap(
                valeurs[colonne], info, ch_type=info.get_channel_types()[0],
                axes=ax, cmap=CARTE_ERD, show=False, contours=0,
                vlim=(-LIMITE_COULEUR, LIMITE_COULEUR), sensors=True,
                names=[n if n in CANAUX_MOTEURS else "" for n in info.ch_names],
            )
            nom_colonne = "Droite moins gauche" if colonne == "difference" else NOMS_CLASSES[colonne]
            ax.set_title(f"{nom_colonne}\n{bande} {BANDES[bande][0]}-{BANDES[bande][1]} Hz",
                         fontsize=9)
    barre = fig.colorbar(image, ax=axes, shrink=0.7)
    barre.set_label("ERD/ERS moyen sur "
                    f"[{FENETRE_ERD[0]}, {FENETRE_ERD[1]}] s (%)")
    fig.suptitle(titre, color="#0b0b0b")
    sauvegarder(fig, nom)


def figure_decours(decours_sujets: list, temps: np.ndarray, nom: str) -> None:
    """Décours temporel de l'ERD mu sur C3 et C4 : main gauche contre main droite, 9 sujets."""
    canaux = decours_sujets[0]["canaux"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, constrained_layout=True)
    for ax, canal in zip(axes, ["C3", "C4"]):
        k = canaux.index(canal)
        _reperes_temps(ax)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        for classe in ["left_hand", "right_hand"]:
            courbes = np.array([d["mu"][classe][k] for d in decours_sujets])
            ax.plot(temps, courbes.T, color=COULEURS_CLASSES[classe], lw=0.6, alpha=0.25)
            ax.plot(temps, courbes.mean(axis=0), color=COULEURS_CLASSES[classe],
                    label=f"{NOMS_CLASSES[classe]} (moyenne)")
        ax.set_title(f"Bande mu (8-12 Hz), électrode {canal}")
        ax.set_xlabel("Temps depuis l'indice (s)")
        ax.grid(axis="y")
    axes[0].set_ylabel("ERD/ERS (%)")
    axes[0].legend(loc="lower left")
    axes[1].annotate("indice", (0, axes[1].get_ylim()[1]), xytext=(3, -10),
                     textcoords="offset points", fontsize=8, color=ENCRE_DISCRETE)
    axes[1].annotate("fin de l'imagerie", (4, axes[1].get_ylim()[1]), xytext=(3, -10),
                     textcoords="offset points", fontsize=8, color=ENCRE_DISCRETE)
    fig.suptitle("Décours de l'ERD mu, 9 sujets (traits fins) et moyenne (trait épais)")
    sauvegarder(fig, nom)


def figure_lateralisation(lat: pd.DataFrame, nom: str) -> None:
    """Indice de latéralisation par sujet et par dérivation : négatif = ERD plus forte du côté controlatéral."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, sharey=True,
                             constrained_layout=True)
    largeur = 0.38
    for i, derivation in enumerate(DERIVATIONS):
        for j, bande in enumerate(BANDES):
            ax = axes[i, j]
            sous = lat[(lat["derivation"] == derivation) & (lat["bande"] == bande)]
            for k, classe in enumerate(["left_hand", "right_hand"]):
                valeurs = sous[sous["classe"] == classe].sort_values("sujet")
                ax.bar(valeurs["sujet"] + (k - 0.5) * largeur, valeurs["indice_lateralisation"],
                       width=largeur - 0.04, color=COULEURS_CLASSES[classe],
                       label=NOMS_CLASSES[classe])
            ax.axhline(0, color=ENCRE_SECONDAIRE, lw=0.8)
            ax.set_xticks(SUJETS)
            ax.grid(axis="y")
            n_attendu = int(sous["controlateral_attendu"].sum())
            ax.set_title(f"{DERIVATIONS[derivation]}, bande {bande} "
                         f"({BANDES[bande][0]}-{BANDES[bande][1]} Hz) : "
                         f"{n_attendu}/{len(sous)} cas controlatéraux", fontsize=9)
            if i == 1:
                ax.set_xlabel("Sujet")
            if j == 0:
                ax.set_ylabel("ERD controlatéral - ipsilatéral\n(points de %)")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Latéralisation par sujet : barre négative = désynchronisation "
                 "plus forte dans l'hémisphère opposé à la main")
    sauvegarder(fig, nom)


def main() -> None:
    fixer_graine()
    mne.set_log_level("ERROR")
    appliquer_style()
    DOSSIER_RESULTATS.mkdir(exist_ok=True)

    print("1. Vérification des marqueurs et de la fenêtre d'imagerie (9 sujets)")
    verification = pd.DataFrame([verifier_marqueurs(s) for s in SUJETS])
    verification.to_csv(DOSSIER_RESULTATS / "phase1_verification_marqueurs.csv", index=False)
    print(verification.to_string(index=False))

    tableaux_erd, tableaux_lat = [], []
    for derivation, nom_derivation in DERIVATIONS.items():
        print(f"\n2. Cartes temps-fréquence et ERD par sujet ({nom_derivation})")
        tableaux, decours_sujets, somme = [], [], None
        for sujet in SUJETS:
            epoques = epoquer_session(sujet, laplacien=(derivation == "laplacien"))
            cartes = erd_temps_frequence(epoques)
            erd = tableau_erd(cartes, sujet)
            tableaux.append(erd)
            decours_sujets.append({"canaux": cartes["left_hand"].ch_names,
                                   "mu": decours_temporel(cartes, "mu")})
            if sujet == 1:
                figure_temps_frequence(
                    cartes, f"Sujet 1 : cartes temps-fréquence, {nom_derivation} (session 1)",
                    f"phase1_tfr_sujet01_{derivation}.png")
                figure_topographies(
                    cartes, erd, f"Sujet 1 : topographies de l'ERD, {nom_derivation} (session 1)",
                    f"phase1_topo_sujet01_{derivation}.png")
            # Moyenne entre sujets : on additionne les cartes d'ERD% déjà normalisées par sujet
            if somme is None:
                somme = {c: t.copy() for c, t in cartes.items()}
            else:
                for c in somme:
                    somme[c].data += cartes[c].data
            print(f"  sujet {sujet} : {len(epoques)} essais")

        moyenne = somme
        for c in moyenne:
            moyenne[c].data /= len(SUJETS)
        erd = pd.concat(tableaux, ignore_index=True)
        erd_moyen = erd.groupby(["classe", "bande", "canal"], as_index=False)["erd_pct"].mean()

        figure_temps_frequence(
            moyenne, f"Moyenne des 9 sujets : cartes temps-fréquence, {nom_derivation} (session 1)",
            f"phase1_tfr_moyenne_{derivation}.png")
        figure_topographies(
            moyenne, erd_moyen,
            f"Moyenne des 9 sujets : topographies de l'ERD, {nom_derivation} (session 1)",
            f"phase1_topo_moyenne_{derivation}.png")
        figure_decours(decours_sujets, moyenne["left_hand"].times,
                       f"phase1_decours_mu_c3_c4_{derivation}.png")

        lat = indices_lateralisation(erd)
        erd.insert(1, "derivation", derivation)
        lat.insert(1, "derivation", derivation)
        tableaux_erd.append(erd)
        tableaux_lat.append(lat)

        print(f"\nERD moyen (9 sujets) sur C3, Cz, C4, {nom_derivation} :")
        print(erd_moyen[erd_moyen["canal"].isin(CANAUX_MOTEURS)]
              .pivot_table(index=["bande", "classe"], columns="canal", values="erd_pct")
              .round(1).to_string())

    erd = pd.concat(tableaux_erd, ignore_index=True).round({"erd_pct": 2})
    erd.to_csv(DOSSIER_RESULTATS / "phase1_erd_par_sujet.csv", index=False)

    print("\n3. Latéralisation controlatérale")
    lat = pd.concat(tableaux_lat, ignore_index=True).round(2)
    lat.to_csv(DOSSIER_RESULTATS / "phase1_lateralisation.csv", index=False)
    figure_lateralisation(lat, "phase1_lateralisation_par_sujet.png")
    resume = lat.groupby(["derivation", "bande", "classe"]).agg(
        cas_controlateraux=("controlateral_attendu", "sum"),
        indice_median=("indice_lateralisation", "median"),
    ).reset_index()
    resume.to_csv(DOSSIER_RESULTATS / "phase1_lateralisation_resume.csv", index=False)
    print(resume.to_string(index=False))
    print(lat.groupby("derivation")["controlateral_attendu"].mean().round(2)
          .rename("part des cas controlatéraux").to_string())


if __name__ == "__main__":
    main()
