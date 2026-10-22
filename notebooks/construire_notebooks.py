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


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = NOTEBOOK_01
    nb.metadata["kernelspec"] = {"name": "bci-imagerie-motrice", "display_name": "Python (.venv bci-imagerie-motrice)", "language": "python"}
    chemin = DOSSIER / "01_exploration_neurophysiologique.ipynb"
    nbf.write(nb, chemin)
    print(f"Notebook écrit : {chemin.name}")


if __name__ == "__main__":
    main()
