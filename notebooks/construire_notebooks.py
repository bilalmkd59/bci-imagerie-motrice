"""Génère les notebooks du projet à partir de cellules définies ici (source unique, versionnable).

Usage (depuis la racine du projet) :
    .venv\\Scripts\\python.exe notebooks\\construire_notebooks.py
    .venv\\Scripts\\python.exe -m nbconvert --to notebook --execute --inplace notebooks\\01_exploration_neurophysiologique.ipynb
"""

from pathlib import Path

import nbformat as nbf

DOSSIER = Path(__file__).resolve().parent


def md(texte: str):
    return nbf.v4.new_markdown_cell(texte.strip())


def code(texte: str):
    return nbf.v4.new_code_cell(texte.strip())


NOTEBOOK_01 = [
    md("""
# 01. Exploration neurophysiologique de l'imagerie motrice

**Question :** quand un sujet *imagine* bouger une main, voit-on une signature mesurable dans l'EEG,
et est-elle latéralisée ?

Tout est calculé sur la **session d'entraînement uniquement** : la session de test reste vierge
pour l'évaluation des phases suivantes. Le code de calcul est dans `src/exploration.py`,
ce notebook ne fait que l'appeler et afficher.
"""),
    code("""
import os, sys
from pathlib import Path
RACINE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(RACINE); sys.path.insert(0, str(RACINE))

import mne
import pandas as pd
from IPython.display import Image, display
from src.chargement import charger_brut, fixer_graine
from src.exploration import epoquer_session, verifier_marqueurs

fixer_graine()
mne.set_log_level("ERROR")
"""),
    md("""
## 1. Structure des données du sujet 1

Chaque session contient 6 runs de 48 essais (12 par classe), enregistrés sur 22 électrodes EEG
et 3 électrodes EOG à 250 Hz.
"""),
    code("""
raw = charger_brut(1)["0train"]["0"]
print(raw)
print("Canaux EEG :", raw.copy().pick("eeg").ch_names)
print("Premières annotations :", list(zip(raw.annotations.onset[:4], raw.annotations.description[:4])))
"""),
    md("""
## 2. Où commence la fenêtre d'imagerie ?

Déroulé officiel d'un essai : croix de fixation et bip à t = 0 s, flèche (indice) à t = 2 s,
imagerie jusqu'à t = 6 s. MOABB déclare `interval = [2, 6]`.

Piège : dans la version installée de MOABB, chaque annotation est placée **sur l'indice** (t = 2 s)
avec une durée de 4 s, et non au début de l'essai. On vérifie donc, contre le fichier `.mat`
officiel de la compétition, le décalage des marqueurs puis l'endroit où commencent réellement
les époques du paradigme (corrélation avec le signal brut à +0 s et +2 s du marqueur).
"""),
    code("""
verification = pd.read_csv("results/phase1_verification_marqueurs.csv")
verification
"""),
    md("""
**Lecture :** décalage constant de +2 s (le marqueur est sur l'indice), étiquettes identiques au fichier
officiel, et l'époque du paradigme coïncide avec le signal brut au marqueur (+0 s), pas à +2 s.
La fenêtre utilisée par MOABB est donc bien [indice, indice + 4 s] = [2, 6] s en temps d'essai.
MOABB affiche simplement les temps de l'époque en « temps d'essai » (2 à 6 s).

## 3. Pourquoi un laplacien de surface ?

Les signaux sont enregistrés contre une référence commune (mastoïde gauche). Par conduction volumique,
une source puissante, comme le rythme alpha visuel à l'arrière de la tête, diffuse sur toutes
les électrodes, y compris C3 et C4, et masque l'ERD motrice locale. Le laplacien de surface
(`mne.preprocessing.compute_current_source_density`) retire à chaque électrode l'activité commune
à ses voisines et ne garde que la source située juste sous elle. Les deux dérivations sont calculées
et comparées en section 7 ; les figures principales utilisent le laplacien.

## 4. Cartes temps-fréquence (sujet 1)

ERD% = (P - R) / R x 100, avec R la puissance de repos sur [-1,5 ; -0,5] s avant l'indice.
Bleu = baisse de puissance (désynchronisation, ERD), rouge = hausse (ERS).
Les lignes blanches marquent 8, 12 et 30 Hz ; les pointillés, l'indice et la fin de l'imagerie.
La tache rouge à 4-8 Hz juste après l'indice est la réponse évoquée visuelle à la flèche, pas de l'imagerie :
c'est pourquoi la fenêtre de mesure de l'ERD commence à 0,5 s.
"""),
    code('display(Image("reports/figures/phase1_tfr_sujet01_laplacien.png"))'),
    md("## 5. Topographies de l'ERD (sujet 1)"),
    code('display(Image("reports/figures/phase1_topo_sujet01_laplacien.png"))'),
    md("""
## 6. Moyenne des 9 sujets

Chaque sujet est d'abord normalisé par sa propre période de repos, puis les cartes d'ERD% sont moyennées.
"""),
    code('display(Image("reports/figures/phase1_tfr_moyenne_laplacien.png"))'),
    code('display(Image("reports/figures/phase1_topo_moyenne_laplacien.png"))'),
    code('display(Image("reports/figures/phase1_decours_mu_c3_c4_laplacien.png"))'),
    md("""
## 7. Latéralisation sujet par sujet, selon la dérivation

Indice = ERD controlatéral - ERD ipsilatéral (points de %). Négatif = désynchronisation plus forte
dans l'hémisphère opposé à la main imaginée, comme attendu.
"""),
    code("""
lat = pd.read_csv("results/phase1_lateralisation.csv")
display(Image("reports/figures/phase1_lateralisation_par_sujet.png"))
lat.pivot_table(index="sujet", columns=["derivation", "bande", "classe"], values="indice_lateralisation")
"""),
    code("""
print("Part des cas (sujet x bande x main) avec latéralisation controlatérale :")
print(lat.groupby("derivation")["controlateral_attendu"].mean().map("{:.0%}".format).to_string())
pd.read_csv("results/phase1_lateralisation_resume.csv")
"""),
]


NOTEBOOK_02 = [
    md("""
# 02. Référence classique : CSP + LDA

**Pipeline :** filtrage 8-30 Hz (MOABB, sur le signal continu) puis, dans un `Pipeline` scikit-learn
réappris à chaque pli : CSP (log-puissance de 6 composantes en binaire, 8 en 4 classes) et LDA.
Fenêtre : 0,5 à 2,5 s après l'indice, fixée a priori.

**Évaluations :**
- intra-session : validation croisée stratifiée en 5 plis, dans chaque session ;
- inter-session : apprentissage sur la session 1, test sur la session 2 (protocole de la compétition).

**Niveau du hasard :** chaque évaluation est refaite 100 fois avec des étiquettes permutées.

Calculs : `src/phase2_csp_lda.py` (ce notebook ne fait que relire les résultats).
"""),
    code("""
import os, sys
from pathlib import Path
RACINE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(RACINE); sys.path.insert(0, str(RACINE))
import pandas as pd
from IPython.display import Image, display
"""),
    md("## 1. Résumé sur les 9 sujets (moyenne et écart-type entre sujets)"),
    code('pd.read_csv("results/phase2_resume.csv").round(3)'),
    md("""
## 2. Scores par sujet

Barres : accuracy. Chiffre : kappa de Cohen, kappa = (accuracy observée - accuracy attendue par hasard)
/ (1 - accuracy attendue). Trait foncé : 95e centile des accuracies obtenues avec étiquettes permutées.
Une barre sous ce trait n'est pas distinguable du hasard.
"""),
    code('display(Image("reports/figures/phase2_scores_par_sujet.png"))'),
    code("""
par_sujet = pd.read_csv("results/phase2_scores_par_sujet.csv")
par_sujet.pivot_table(index="sujet", columns=["tache", "evaluation"], values="kappa").round(3)
"""),
    md("""
## 3. Niveau du hasard et p-valeurs

Avec 100 permutations, la plus petite p-valeur possible est 1/101, soit environ 0,0099.
Le seuil de hasard est plus élevé en inter-session : un seul ensemble de test, sans moyenne sur des plis,
donne une distribution de scores aléatoires plus étalée.
"""),
    code("""
scores = pd.read_csv("results/phase2_scores.csv")
scores[scores["p_valeur"] >= 0.05][["sujet", "tache", "evaluation", "session", "accuracy",
                                    "hasard_accuracy_p95", "p_valeur"]]
"""),
    md("## 4. Matrices de confusion (4 classes, cumulées sur les 9 sujets)"),
    code('display(Image("reports/figures/phase2_confusion_4classes.png"))'),
    md("""
## 5. Filtres et motifs CSP

- Un **filtre** est le jeu de poids appliqué aux 22 électrodes pour fabriquer une composante : c'est ce que
  le modèle *utilise*. Il annule aussi le bruit commun, d'où son aspect irrégulier, difficile à interpréter.
- Un **motif** (pattern) est la carte de la source que la composante capte, telle qu'elle se projette sur le scalp :
  c'est ce qu'il faut *interpréter* physiologiquement (Haufe et al., 2014).
- Le signe d'une composante est arbitraire : seule la forme de la carte compte.

Sujet 3 (meilleur en validation croisée sur la session 1), puis sujet 5 (le moins bon) et sujet 1.
"""),
    code('display(Image("reports/figures/phase2_csp_sujet03.png"))'),
    code('display(Image("reports/figures/phase2_csp_sujet05.png"))'),
    code('display(Image("reports/figures/phase2_csp_sujet01.png"))'),
]

NOTEBOOKS = {
    "01_exploration_neurophysiologique.ipynb": NOTEBOOK_01,
    "02_csp_lda.ipynb": NOTEBOOK_02,
}


def main() -> None:
    for nom, cellules in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cellules
        nb.metadata["kernelspec"] = {"name": "bci-imagerie-motrice",
                                     "display_name": "Python (.venv bci-imagerie-motrice)",
                                     "language": "python"}
        nbf.write(nb, DOSSIER / nom)
        print(f"Notebook écrit : {nom}")


if __name__ == "__main__":
    main()
