"""Pipelines de classification.

Chaque pipeline est un objet scikit-learn neuf, créé par une fonction « fabrique » :
tout ce qui apprend sur les données (filtres CSP, classifieur) est réappris à chaque
pli de validation, ce qui empêche toute fuite d'information du test vers l'apprentissage.
"""

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

# Nombre de composantes CSP, fixé a priori (non optimisé sur les données) :
# 3 paires en binaire, 2 composantes par classe en 4 classes.
N_COMPOSANTES_CSP = {"binaire": 6, "4classes": 8}


def csp_lda(tache: str) -> Pipeline:
    """CSP (puissance log des composantes) suivi d'une analyse discriminante linéaire."""
    # En binaire, ordre « alternate » : les composantes vont par paires extrêmes,
    # (variance maximale pour une classe, puis pour l'autre), ce qui les rend interprétables.
    # En multiclasse, MNE impose l'ordre par information mutuelle avec les classes.
    ordre = "alternate" if tache == "binaire" else "mutual_info"
    return Pipeline([
        ("csp", CSP(n_components=N_COMPOSANTES_CSP[tache], reg=None, log=True,
                    component_order=ordre)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
