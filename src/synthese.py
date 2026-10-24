"""Phase 6 : synthèse générée automatiquement à partir des CSV de results/.

Usage (depuis la racine du projet, après les phases 2 à 5) :
    .venv\\Scripts\\python.exe -m src.synthese

Aucun chiffre n'est écrit à la main : tous sont lus dans les CSV produits par les scripts
des phases précédentes. Produit :
    results/synthese.md                    synthèse complète
    results/synthese_tableau.csv           tableau comparatif (moyenne ± écart-type, 9 sujets)
    reports/figures/synthese_methodes.png  figure récapitulative
    README.md                              tableau de résultats mis à jour entre deux balises
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import ENCRE_DISCRETE, ENCRE_SECONDAIRE, RACINE, appliquer_style, sauvegarder
from src.rapport_pedagogique import GLOSSAIRE, partie_explicative

RESULTATS = RACINE / "results"
TACHES = {"4classes": "4 classes", "binaire": "Main gauche contre main droite"}
PROTOCOLES = {"intra": "Intra-session", "inter": "Inter-session",
              "loso": "Sujet jamais vu"}
# Ordre d'affichage des méthodes et variantes
METHODES = [
    "CSP+LDA", "TS+LR", "MDM", "EEGNet",
    "CSP+LDA, sans recentrage", "TS+LR, sans recentrage", "MDM, sans recentrage",
    "EEGNet, sans alignement",
    "TS+LR, recentrage (réf. session 1)", "MDM, recentrage (réf. session 1)",
    "EEGNet, alignement (réf. session 1)",
    "TS+LR, recentrage transductif", "MDM, recentrage transductif",
]
COULEURS = {"CSP+LDA": "#2a78d6", "TS+LR": "#eb6834", "MDM": "#1baf7a", "EEGNet": "#e87ba4"}
BALISE_DEBUT, BALISE_FIN = "<!-- RESULTATS:DEBUT -->", "<!-- RESULTATS:FIN -->"


def nombre(valeur: float, decimales: int = 3) -> str:
    """Format français : virgule décimale et vrai signe moins."""
    return f"{valeur:.{decimales}f}".replace(".", ",").replace("-", "−")


def p_valeur(valeur: float) -> str:
    return nombre(valeur, 4)


# ---------------------------------------------------------------- lecture des scores

def scores_par_sujet() -> pd.DataFrame:
    """Un kappa et une accuracy par (méthode, tâche, protocole, sujet), à partir de tous les CSV."""
    morceaux = []
    phase3 = pd.read_csv(RESULTATS / "phase3_scores.csv")
    morceaux.append(phase3.rename(columns={"evaluation": "protocole"}))
    for protocole in ["intra", "inter"]:
        chemin = RESULTATS / f"phase5_scores_{protocole}.csv"
        if chemin.exists():
            morceaux.append(pd.read_csv(chemin).rename(columns={"evaluation": "protocole"}))
    intra_inter = pd.concat(morceaux, ignore_index=True)
    # Intra-session : moyenne des deux sessions pour avoir un score par sujet
    intra_inter = (intra_inter.groupby(["methode", "tache", "protocole", "sujet"], as_index=False)
                   [["accuracy", "kappa"]].mean())

    noms_alignement = {"aucun": "sans recentrage", "calibration": "recentrage (réf. session 1)",
                       "transductif": "recentrage transductif"}
    loso = pd.read_csv(RESULTATS / "phase4_scores.csv")
    loso["methode"] = loso["methode"] + ", " + loso["alignement"].map(noms_alignement)
    morceaux_loso = [loso]
    chemin = RESULTATS / "phase5_scores_loso.csv"
    if chemin.exists():
        eeg = pd.read_csv(chemin)
        eeg["methode"] = "EEGNet, " + eeg["alignement"].map(
            {"aucun": "sans alignement", "calibration": "alignement (réf. session 1)"})
        morceaux_loso.append(eeg)
    loso = pd.concat(morceaux_loso, ignore_index=True).assign(protocole="loso")
    colonnes = ["methode", "tache", "protocole", "sujet", "accuracy", "kappa"]
    return pd.concat([intra_inter[colonnes], loso[colonnes]], ignore_index=True)


def tableau_comparatif(scores: pd.DataFrame) -> pd.DataFrame:
    tableau = scores.groupby(["tache", "protocole", "methode"]).agg(
        accuracy_moyenne=("accuracy", "mean"), accuracy_ecart_type=("accuracy", "std"),
        kappa_moyen=("kappa", "mean"), kappa_ecart_type=("kappa", "std"),
        n_sujets=("sujet", "nunique")).reset_index()
    tableau["ordre"] = tableau["methode"].map({m: i for i, m in enumerate(METHODES)})
    return tableau.sort_values(["tache", "protocole", "ordre"]).drop(columns="ordre")


# ---------------------------------------------------------------- tableaux markdown

def cellule(ligne: pd.Series) -> str:
    return (f"{nombre(ligne['kappa_moyen'])} ± {nombre(ligne['kappa_ecart_type'])} | "
            f"{nombre(ligne['accuracy_moyenne'])} ± {nombre(ligne['accuracy_ecart_type'])}")


def tableau_markdown(tableau: pd.DataFrame, tache: str, protocole: str) -> str:
    sous = tableau[(tableau["tache"] == tache) & (tableau["protocole"] == protocole)]
    lignes = ["| Méthode | Kappa (moyenne ± écart-type) | Accuracy (moyenne ± écart-type) |",
              "|---|---|---|"]
    meilleur = sous["kappa_moyen"].max()
    for _, ligne in sous.iterrows():
        nom = f"**{ligne['methode']}**" if ligne["kappa_moyen"] == meilleur else ligne["methode"]
        lignes.append(f"| {nom} | {cellule(ligne)} |")
    return "\n".join(lignes)


def tableau_resume_readme(tableau: pd.DataFrame) -> str:
    """Tableau compact (kappa moyen) pour le README : méthodes principales, 3 protocoles."""
    lignes = ["| Tâche | Méthode | Intra-session | Inter-session | Sujet jamais vu* |",
              "|---|---|---|---|---|"]
    variantes_loso = {"CSP+LDA": "CSP+LDA, sans recentrage",
                      "TS+LR": "TS+LR, recentrage (réf. session 1)",
                      "MDM": "MDM, recentrage (réf. session 1)",
                      "EEGNet": "EEGNet, alignement (réf. session 1)"}
    for tache, nom_tache in TACHES.items():
        for methode, variante in variantes_loso.items():
            valeurs = []
            for protocole, nom_methode in [("intra", methode), ("inter", methode),
                                           ("loso", variante)]:
                ligne = tableau[(tableau["tache"] == tache) & (tableau["protocole"] == protocole)
                                & (tableau["methode"] == nom_methode)]
                valeurs.append(f"{nombre(ligne['kappa_moyen'].iloc[0])} ± "
                               f"{nombre(ligne['kappa_ecart_type'].iloc[0])}"
                               if not ligne.empty else "non calculé")
            lignes.append(f"| {nom_tache} | {methode} | " + " | ".join(valeurs) + " |")
    return "\n".join(lignes)


# ---------------------------------------------------------------- sections de la synthèse

def section_hasard() -> str:
    par_sujet = pd.read_csv(RESULTATS / "phase2_scores_par_sujet.csv")
    lignes = ["| Tâche | Évaluation | Hasard théorique | 95e centile des permutations (moyenne des sujets) "
              "| Sujets au-dessus du hasard (p < 0,05) |", "|---|---|---|---|---|"]
    for tache, nom_tache in TACHES.items():
        for evaluation in ["intra", "inter"]:
            sous = par_sujet[(par_sujet["tache"] == tache) & (par_sujet["evaluation"] == evaluation)]
            theorique = "50 %" if tache == "binaire" else "25 %"
            lignes.append(f"| {nom_tache} | {PROTOCOLES[evaluation]} | {theorique} | "
                          f"{nombre(sous['hasard_accuracy_p95'].mean())} | "
                          f"{int(sous['au_dessus_du_hasard'].sum())}/9 |")
    return "\n".join(lignes)


def section_sujets(scores: pd.DataFrame) -> str:
    """Meilleurs et pires sujets, protocole de la compétition (4 classes, inter-session)."""
    sous = scores[(scores["tache"] == "4classes") & (scores["protocole"] == "inter")]
    kappa = sous.pivot(index="sujet", columns="methode", values="kappa")
    methodes = [m for m in ["CSP+LDA", "TS+LR", "MDM", "EEGNet"] if m in kappa.columns]
    kappa = kappa[methodes]
    kappa["moyenne des méthodes"] = kappa.mean(axis=1)
    kappa = kappa.sort_values("moyenne des méthodes", ascending=False)
    rangs = kappa[methodes].rank(ascending=False)
    accord = rangs.corr(method="spearman").where(~np.eye(len(methodes), dtype=bool)).stack().mean()

    entete = "| Sujet | " + " | ".join(kappa.columns) + " |"
    lignes = [entete, "|" + "---|" * (len(kappa.columns) + 1)]
    for sujet, ligne in kappa.iterrows():
        lignes.append(f"| {sujet} | " + " | ".join(nombre(v) for v in ligne) + " |")
    meilleurs = ", ".join(str(s) for s in kappa.index[:3])
    pires = ", ".join(str(s) for s in kappa.index[-3:])
    ecart_sujets = kappa["moyenne des méthodes"].max() - kappa["moyenne des méthodes"].min()
    moyennes_methodes = kappa[methodes].mean()
    ecart_methodes = moyennes_methodes.max() - moyennes_methodes.min()
    # Les qualificatifs dépendent des valeurs calculées, pour rester vrais si les chiffres changent
    comparaison = ("bien plus grand que" if ecart_sujets > 2 * ecart_methodes
                   else "comparable à")
    stabilite = ("très stable" if accord >= 0.7 else "assez stable" if accord >= 0.4
                 else "peu stable")
    texte = (f"Meilleurs sujets : **{meilleurs}** ; sujets les plus difficiles : **{pires}**. "
             f"L'écart de kappa moyen entre le meilleur et le moins bon sujet est de "
             f"{nombre(ecart_sujets)}, {comparaison} l'écart entre la meilleure et la moins bonne "
             f"méthode ({nombre(ecart_methodes)}). "
             f"Le classement des sujets est {stabilite} d'une méthode à l'autre "
             f"(corrélation de Spearman moyenne entre méthodes : {nombre(accord, 2)}).")
    if accord >= 0.7 and ecart_sujets > 2 * ecart_methodes:
        texte += (" La difficulté tient donc au sujet plus qu'à l'algorithme : c'est la signature "
                  "de l'« illettrisme BCI », une partie des utilisateurs ne produisant pas de "
                  "désynchronisation assez nette pour être décodée.")
    return "\n".join(lignes) + "\n\n" + texte


def section_tests() -> str:
    lignes = []
    w3 = pd.read_csv(RESULTATS / "phase3_wilcoxon.csv")
    for tache in TACHES:
        r = w3[(w3["tache"] == tache) & (w3["evaluation"] == "inter")
               & (w3["methode_a"] == "TS+LR") & (w3["methode_b"] == "CSP+LDA")].iloc[0]
        lignes.append(
            f"- **TS+LR contre CSP+LDA**, {TACHES[tache].lower()}, inter-session : "
            f"différence de kappa {nombre(r['difference_moyenne'])}, "
            f"{int(r['sujets_a_meilleur'])}/9 sujets améliorés, p = {p_valeur(r['p_valeur'])}, "
            f"p corrigée (Holm, 12 tests) = {p_valeur(r['p_valeur_holm'])}.")
    r = w3[(w3["tache"] == "4classes") & (w3["evaluation"] == "inter")
           & (w3["methode_a"] == "TS+LR") & (w3["methode_b"] == "MDM")].iloc[0]
    lignes.append(f"- **TS+LR contre MDM**, 4 classes, inter-session : {int(r['sujets_a_meilleur'])}/9 "
                  f"sujets améliorés, p = {p_valeur(r['p_valeur'])}, "
                  f"p corrigée = {p_valeur(r['p_valeur_holm'])}.")

    w4 = pd.read_csv(RESULTATS / "phase4_wilcoxon.csv")
    for a, b, intitule in [("aucun", "propre au sujet", "perte quand le sujet n'a jamais été vu"),
                           ("calibration", "aucun", "gain du recentrage (référence : session 1)"),
                           ("transductif", "aucun", "gain du recentrage transductif")]:
        r = w4[(w4["tache"] == "4classes") & (w4["methode"] == "TS+LR")
               & (w4["alignement_a"] == a) & (w4["alignement_b"] == b)].iloc[0]
        lignes.append(f"- **TS+LR, 4 classes, {intitule}** : différence de kappa "
                      f"{nombre(r['difference_moyenne'])}, {int(r['sujets_a_meilleur'])}/9 sujets "
                      f"en faveur de « {a} », p = {p_valeur(r['p_valeur'])}, "
                      f"p corrigée (Holm, 12 tests) = {p_valeur(r['p_valeur_holm'])}.")
    recuperation = pd.read_csv(RESULTATS / "phase4_recuperation.csv")
    for _, r in recuperation[recuperation["tache"] == "4classes"].iterrows():
        lignes.append(f"- Part de la perte inter-sujets récupérée par le recentrage "
                      f"({r['alignement']}, 4 classes, TS+LR) : "
                      f"{nombre(100 * r['part_de_la_perte_recuperee'], 0)} %.")

    chemin = RESULTATS / "phase5_wilcoxon.csv"
    if chemin.exists():
        w5 = pd.read_csv(chemin)
        for protocole in ["intra", "inter", "loso"]:
            r = w5[(w5["protocole"] == protocole) & (w5["tache"] == "4classes")
                   & (w5["methode_b"] == "CSP+LDA")]
            if r.empty:
                continue
            r = r.iloc[0]
            lignes.append(
                f"- **EEGNet contre CSP+LDA**, 4 classes, {PROTOCOLES[protocole].lower()} : "
                f"différence de kappa {nombre(r['difference_moyenne'])}, "
                f"{int(r['sujets_a_meilleur'])}/9 sujets en faveur d'EEGNet, "
                f"p = {p_valeur(r['p_valeur'])}, p corrigée (Holm, {len(w5)} tests) = "
                f"{p_valeur(r['p_valeur_holm'])}.")
    return "\n".join(lignes)


LIMITES = """\
- **9 sujets seulement.** Avec 9 sujets, la plus petite p-valeur possible au test de Wilcoxon est
  2/512, soit environ 0,004 : après correction pour comparaisons multiples, seules les différences
  observées chez les 9 sujets peuvent être déclarées significatives. Plusieurs tendances (TS+LR
  meilleur que CSP+LDA, EEGNet meilleur que CSP+LDA en inter-sujets) restent donc non démontrées.
- **Un seul jeu de données** (BCI Competition IV 2a) : les conclusions demandent à être confirmées
  sur d'autres jeux de données d'imagerie motrice disponibles dans MOABB.
- **Analyse hors ligne.** Le filtrage passe-bande est appliqué sur l'enregistrement continu, avec un
  filtre non causal (qui utilise des échantillons futurs) : c'est sans fuite d'étiquettes, mais un
  système en temps réel utiliserait un filtre causal.
- **Essais marqués comme artefacts conservés.** MOABB renvoie les 288 essais de chaque session, y compris
  ceux que les organisateurs ont signalés comme contaminés ; les scores incluent donc ces essais.
- **Réglages fixés a priori, sans optimisation.** Fenêtre (0,5 à 2,5 s), bande (8-30 Hz), nombre de
  composantes CSP, hyperparamètres d'EEGNet : choisis avant toute évaluation. C'est un choix de rigueur
  (aucune sélection sur le test), mais toutes les méthodes pourraient gagner à un réglage par validation
  croisée imbriquée sur les données d'apprentissage.
- **Niveau du hasard par permutation calculé en phase 2 seulement** (100 permutations, coût de calcul) ;
  les phases suivantes comparent les méthodes par tests appariés.
- **EEGNet : une seule graine par entraînement, matériel mixte.** Le diagnostic montre jusqu'à 0,25 de
  kappa d'écart entre graines sur un même sujet ; l'inter-session a été calculé sur CPU (PyTorch 2.14),
  l'intra-session et le protocole inter-sujets sur GPU Colab (PyTorch 2.11). Deux exécutions sur CPU
  peuvent différer d'un essai (calculs parallèles non strictement déterministes).
- **Recentrage transductif optimiste.** Il calcule la référence du nouveau sujet sur ses essais de test
  (sans étiquettes) ; la variante réaliste est celle dont la référence est la session 1."""


# ---------------------------------------------------------------- figure

def figure(tableau: pd.DataFrame) -> None:
    """Kappa moyen ± écart-type : 4 méthodes x 3 protocoles, variante réaliste en inter-sujets."""
    variantes_loso = {"CSP+LDA": "CSP+LDA, sans recentrage",
                      "TS+LR": "TS+LR, recentrage (réf. session 1)",
                      "MDM": "MDM, recentrage (réf. session 1)",
                      "EEGNet": "EEGNet, alignement (réf. session 1)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, constrained_layout=True)
    for ax, (tache, nom_tache) in zip(axes, TACHES.items()):
        for k, (methode, variante) in enumerate(variantes_loso.items()):
            for j, protocole in enumerate(PROTOCOLES):
                nom = variante if protocole == "loso" else methode
                ligne = tableau[(tableau["tache"] == tache) & (tableau["protocole"] == protocole)
                                & (tableau["methode"] == nom)]
                if ligne.empty:
                    continue
                x = j + (k - 1.5) * 0.17
                ax.errorbar(x, ligne["kappa_moyen"].iloc[0], yerr=ligne["kappa_ecart_type"].iloc[0],
                            fmt="o", ms=7, capsize=3, lw=1.2, color=COULEURS[methode],
                            label=methode if j == 0 else None)
        ax.axhline(0, color=ENCRE_DISCRETE, lw=0.8, ls="--")
        ax.set_xticks(range(len(PROTOCOLES)), [
            "Intra-session", "Inter-session",
            "Sujet jamais vu\n(CSP+LDA sans recentrage,\nautres : réf. session 1)"], fontsize=8)
        ax.set_title(nom_tache, fontsize=10)
        ax.grid(axis="y")
    axes[0].set_ylabel("Kappa de Cohen, moyenne ± écart-type (9 sujets)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Synthèse : quatre méthodes, trois protocoles d'évaluation",
                 color=ENCRE_SECONDAIRE)
    sauvegarder(fig, "synthese_methodes.png")


# ---------------------------------------------------------------- assemblage

def mettre_a_jour_readme(tableau_readme: str) -> None:
    chemin = RACINE / "README.md"
    if not chemin.exists():
        return
    texte = chemin.read_text(encoding="utf-8")
    if BALISE_DEBUT not in texte or BALISE_FIN not in texte:
        return
    avant, reste = texte.split(BALISE_DEBUT, 1)
    _, apres = reste.split(BALISE_FIN, 1)
    chemin.write_text(f"{avant}{BALISE_DEBUT}\n{tableau_readme}\n{BALISE_FIN}{apres}",
                      encoding="utf-8")


def main() -> None:
    appliquer_style()
    scores = scores_par_sujet()
    tableau = tableau_comparatif(scores)
    tableau.round(4).to_csv(RESULTATS / "synthese_tableau.csv", index=False)
    figure(tableau)

    sections = ["# Interface cerveau-machine par imagerie motrice : rapport de synthèse",
                "",
                "*Projet de Bilal Mokeddem. Document généré automatiquement par `src/synthese.py` à partir "
                "des fichiers de `results/` : aucun chiffre n'est saisi à la main. Sauf mention contraire, "
                "les valeurs sont des moyennes sur les 9 personnes du jeu BCI Competition IV 2a.*",
                "",
                "![Synthèse](../reports/figures/synthese_methodes.png)",
                ""]
    sections += partie_explicative(tableau)
    sections += ["", "## 4. Tableaux comparatifs complets", "",
                 "Moyenne ± écart-type sur les 9 personnes. L'écart-type mesure la dispersion entre "
                 "personnes : un écart-type élevé signifie que la méthode marche très bien chez certaines "
                 "et mal chez d'autres. La meilleure méthode de chaque tableau est en gras. "
                 "Pour le protocole « sujet jamais vu », « réf. session 1 » désigne le recentrage réaliste "
                 "et « transductif » la variante optimiste (voir phase 4)."]
    for tache, nom_tache in TACHES.items():
        for protocole, nom_protocole in PROTOCOLES.items():
            sections += ["", f"### {nom_tache}, {nom_protocole.lower()}", "",
                         tableau_markdown(tableau, tache, protocole)]
    sections += ["", "## 5. Niveau du hasard (CSP+LDA, 100 permutations des étiquettes)", "",
                 "Le « 95e centile » est le score que le hasard ne dépasse que 5 fois sur 100 : "
                 "un vrai score au-dessus de cette valeur est significatif.", "",
                 section_hasard(),
                 "", "## 6. Meilleures et moins bonnes personnes (4 classes, inter-session, kappa)", "",
                 section_sujets(scores),
                 "", "## 7. Tests statistiques principaux (Wilcoxon apparié, 9 personnes)", "",
                 "Lecture : « 7/9 sujets améliorés » signifie que la première méthode fait mieux chez 7 "
                 "personnes sur 9. Une p corrigée inférieure à 0,05 indique une différence significative "
                 "malgré le nombre de tests réalisés.", "",
                 section_tests(),
                 "", "## 8. Limites", "", LIMITES, "", GLOSSAIRE, ""]
    (RESULTATS / "synthese.md").write_text("\n".join(sections), encoding="utf-8")
    mettre_a_jour_readme(tableau_resume_readme(tableau))
    print("Synthèse écrite : results/synthese.md")
    print(tableau_resume_readme(tableau))


if __name__ == "__main__":
    main()
