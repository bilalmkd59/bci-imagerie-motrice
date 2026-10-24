"""Recentrage riemannien des covariances, pour la généralisation entre sujets.

Principe (Zanini et al., 2018) : chaque sujet a sa propre « position » sur la variété des
covariances (anatomie, impédance des électrodes, placement du casque). On ramène chaque
groupe d'essais (un sujet, une session) sur l'identité :

    C  ->  M^(-1/2) C M^(-1/2)

où M est la moyenne riemannienne des covariances du groupe. Après recentrage, tous les groupes
ont la même moyenne (l'identité) : ce qui reste, ce sont les écarts *relatifs* à l'état moyen du
sujet, et ces écarts (une ERD à gauche ou à droite) se ressemblent davantage d'un sujet à l'autre.

Le calcul de M n'utilise aucune étiquette. Il est réalisé à l'intérieur de l'estimateur, qui est
recréé et réappris à chaque pli : aucune référence n'est calculée en dehors du pli.
"""

from typing import Callable

import numpy as np
from pyriemann.classification import MDM
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def recentrer(covariances: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Applique C -> M^(-1/2) C M^(-1/2) à chaque covariance (transport de M vers l'identité)."""
    blanchiment = invsqrtm(reference)
    return blanchiment @ covariances @ blanchiment


def ts_lr_sur_covariances() -> Pipeline:
    """Espace tangent + régression logistique, appliqués à des covariances déjà estimées."""
    return Pipeline([("espace_tangent", TangentSpace(metric="riemann")),
                     ("regression_logistique", LogisticRegression(max_iter=1000))])


def mdm_sur_covariances() -> MDM:
    """MDM appliqué à des covariances déjà estimées."""
    return MDM(metric="riemann")


class ClassifieurRecentre(BaseEstimator, ClassifierMixin):
    """Recentre chaque groupe d'apprentissage sur l'identité, puis apprend un classifieur.

    Ce n'est pas un simple Pipeline scikit-learn car le recentrage a besoin de savoir à quel
    groupe (sujet, session) appartient chaque essai, information qu'un Pipeline ne transmet
    pas à l'étape transform. L'estimateur reste autonome : il est créé neuf à chaque pli et
    tout ce qu'il apprend (moyennes de groupes, classifieur) reste à l'intérieur.

    Paramètres
    ----------
    fabrique_classifieur : fonction sans argument renvoyant un classifieur sur covariances.
    """

    def __init__(self, fabrique_classifieur: Callable):
        self.fabrique_classifieur = fabrique_classifieur

    def fit(self, covariances: np.ndarray, y: np.ndarray, groupes: np.ndarray):
        recentrees = np.empty_like(covariances)
        for groupe in np.unique(groupes):
            masque = groupes == groupe
            # Moyenne du groupe calculée sans étiquette
            recentrees[masque] = recentrer(covariances[masque], mean_riemann(covariances[masque]))
        self.classifieur_ = self.fabrique_classifieur().fit(recentrees, y)
        self.classes_ = self.classifieur_.classes_
        return self

    def predict(self, covariances: np.ndarray, covariances_reference: np.ndarray) -> np.ndarray:
        """Prédit les essais d'un nouveau sujet, recentrés par la moyenne de `covariances_reference`.

        `covariances_reference` sont des essais non étiquetés du même sujet : soit une session
        de calibration distincte (aucune donnée de test utilisée), soit les essais de test
        eux-mêmes (variante transductive, plus optimiste).
        """
        reference = mean_riemann(covariances_reference)
        return self.classifieur_.predict(recentrer(covariances, reference))
