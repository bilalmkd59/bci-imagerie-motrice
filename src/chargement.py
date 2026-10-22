"""Chargement du jeu de données BCI Competition IV 2a (BNCI2014_001) via MOABB.

Les fichiers bruts sont téléchargés une seule fois dans le cache MNE
(par défaut ~/mne_data), en dehors du dépôt git.
"""

import random
import warnings

import numpy as np
from moabb.datasets import BNCI2014_001

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
