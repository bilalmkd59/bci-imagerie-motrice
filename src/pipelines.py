"""Pipelines de classification.

Chaque pipeline est un objet scikit-learn neuf, créé par une fonction « fabrique » :
tout ce qui apprend sur les données (filtres CSP, moyenne riemannienne de référence,
classifieur) est réappris à chaque pli de validation, ce qui empêche toute fuite
d'information du test vers l'apprentissage.

Toutes les fabriques prennent la tâche ("binaire" ou "4classes") en argument,
même celles qui ne s'en servent pas, pour pouvoir être appelées de la même façon.
"""

from mne.decoding import CSP
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Nombre de composantes CSP, fixé a priori (non optimisé sur les données) :
# 3 paires en binaire, 2 composantes par classe en 4 classes.
N_COMPOSANTES_CSP = {"binaire": 6, "4classes": 8}

# Estimateur de covariance : OAS (Oracle Approximating Shrinkage) rétrécit la covariance
# empirique vers une matrice diagonale. Avec 501 échantillons pour 22 canaux, la covariance
# empirique est déjà inversible, mais le rétrécissement la rend plus stable d'un essai à l'autre.
ESTIMATEUR_COVARIANCE = "oas"


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


def ts_lr(tache: str) -> Pipeline:
    """Covariances, projection dans l'espace tangent, puis régression logistique.

    TangentSpace apprend la moyenne riemannienne des covariances d'apprentissage et
    « déplie » la variété autour de ce point : chaque covariance 22 x 22 devient un vecteur
    de 22 x 23 / 2 = 253 coordonnées, sur lesquelles un classifieur linéaire classique s'applique.
    La projection blanchit déjà les données par la moyenne de référence, d'où l'absence de
    standardisation supplémentaire.
    """
    return Pipeline([
        ("covariances", Covariances(estimator=ESTIMATEUR_COVARIANCE)),
        ("espace_tangent", TangentSpace(metric="riemann")),
        ("regression_logistique", LogisticRegression(max_iter=1000)),
    ])


def mdm(tache: str) -> Pipeline:
    """Minimum Distance to Mean : chaque essai va à la classe dont la moyenne riemannienne est la plus proche."""
    return Pipeline([
        ("covariances", Covariances(estimator=ESTIMATEUR_COVARIANCE)),
        ("mdm", MDM(metric="riemann")),
    ])


METHODES = {"CSP+LDA": csp_lda, "TS+LR": ts_lr, "MDM": mdm}
