"""Chargement du jeu de données BCI Competition IV 2a (BNCI2014_001) via MOABB.

Les fichiers bruts sont téléchargés une seule fois dans le cache MNE
(par défaut ~/mne_data), en dehors du dépôt git.
"""

import random
import warnings

import mne
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery, MotorImagery

GRAINE = 42

# Avertissement interne à MOABB (nom de montage déprécié dans MNE), sans effet sur nos données.
warnings.filterwarnings("ignore", message="Montage name 'standard_1005' is deprecated")


def fixer_graine(graine: int = GRAINE) -> None:
    """Fixe les graines aléatoires de Python et NumPy pour la reproductibilité."""
    random.seed(graine)
    np.random.seed(graine)


def charger_dataset() -> BNCI2014_001:
    """Renvoie l'objet MOABB décrivant le jeu de données (rien n'est téléchargé ici)."""
    return BNCI2014_001()


def charger_brut(sujet: int) -> dict:
    """Charge les enregistrements continus (objets mne.io.Raw) d'un sujet.

    Renvoie un dictionnaire imbriqué {session: {run: Raw}}, tel que fourni
    par MOABB. Le téléchargement a lieu au premier appel.
    """
    dataset = charger_dataset()
    return dataset.get_data(subjects=[sujet])[sujet]


# Fenêtre de classification, en secondes après l'indice. Elle est fixée a priori :
# elle évite la réponse évoquée visuelle des 500 premières ms et couvre le cœur de l'ERD.
TMIN_CLASSIF, TMAX_CLASSIF = 0.5, 2.5
FMIN, FMAX = 8, 30  # bandes mu et beta


def charger_epoques(sujet: int, tache: str, avec_epoques: bool = False):
    """Renvoie les essais filtrés en 8-30 Hz d'un sujet, pour les deux sessions.

    tache : "binaire" (main gauche contre main droite) ou "4classes".
    Le filtrage passe-bande est appliqué par MOABB sur le signal continu : il ne dépend
    d'aucune étiquette et n'apprend rien sur les données, il n'y a donc pas de fuite.

    Renvoie (X, y, meta) avec X de forme (essais, 22 canaux, échantillons), y les noms
    de classes et meta un DataFrame (colonne "session" : "0train" ou "1test").
    Avec `avec_epoques=True`, renvoie aussi l'objet mne.Epochs (utile pour les topographies).
    """
    reglages = dict(fmin=FMIN, fmax=FMAX, tmin=TMIN_CLASSIF, tmax=TMAX_CLASSIF)
    if tache == "binaire":
        paradigme = LeftRightImagery(**reglages)
    elif tache == "4classes":
        paradigme = MotorImagery(n_classes=4, **reglages)
    else:
        raise ValueError(f"Tâche inconnue : {tache}")
    with mne.utils.use_log_level("ERROR"):
        epoques, y, meta = paradigme.get_data(charger_dataset(), [sujet], return_epochs=True)
    X = epoques.get_data(copy=True)
    y = np.asarray(y)
    if avec_epoques:
        return X, y, meta, epoques
    return X, y, meta
