"""EEGNet (Lawhern et al., 2018) via braindecode, emballé comme un classifieur scikit-learn.

L'emballage permet d'évaluer EEGNet avec exactement la même boucle, les mêmes essais et les
mêmes plis que les méthodes classiques. Tous les choix d'entraînement sont fixés a priori :
pas d'arrêt précoce (il faudrait un ensemble de validation, et le prendre sur le test serait
une fuite), nombre d'époques fixe et taux d'apprentissage décroissant en cosinus.
"""

import numpy as np
import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from braindecode.util import set_random_seeds
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import LRScheduler

from src.chargement import GRAINE

FREQUENCE_ECHANTILLONNAGE = 250
N_EPOQUES = 200            # protocoles intra- et inter-session (une session, 144 à 288 essais)
N_EPOQUES_INTER_SUJETS = 100  # 16 fois plus d'essais par époque en leave-one-subject-out


class ClassifieurEEGNet(BaseEstimator, ClassifierMixin):
    """EEGNet appris de zéro à chaque appel de fit (un réseau neuf par pli)."""

    def __init__(self, n_epoques: int = N_EPOQUES, taux_apprentissage: float = 1e-3,
                 taille_lot: int = 64, graine: int = GRAINE, appareil: str = "cpu"):
        self.n_epoques = n_epoques
        self.taux_apprentissage = taux_apprentissage
        self.taille_lot = taille_lot
        self.graine = graine
        self.appareil = appareil

    @staticmethod
    def _preparer(X: np.ndarray) -> np.ndarray:
        """Volts vers microvolts, en float32. Mise à l'échelle fixe : rien n'est appris sur les données."""
        return (X * 1e6).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # skorch attend des étiquettes entières : on encode, et on décodera à la prédiction
        self.encodeur_ = LabelEncoder().fit(y)
        self.classes_ = self.encodeur_.classes_
        set_random_seeds(self.graine, cuda=self.appareil == "cuda")
        self.reseau_ = EEGClassifier(
            EEGNet,
            module__n_chans=X.shape[1],
            module__n_outputs=len(self.classes_),
            module__n_times=X.shape[2],
            # Noyau temporel d'une demi-seconde, règle de l'article original (64 à 128 Hz)
            module__kernel_length=FREQUENCE_ECHANTILLONNAGE // 2,
            optimizer=torch.optim.AdamW,
            lr=self.taux_apprentissage,
            batch_size=self.taille_lot,
            max_epochs=self.n_epoques,
            callbacks=[("taux", LRScheduler(policy="CosineAnnealingLR",
                                             T_max=max(1, self.n_epoques - 1)))],
            train_split=None,  # aucune validation interne : pas d'arrêt précoce
            device=self.appareil,
            verbose=0,
        )
        self.reseau_.fit(self._preparer(X), self.encodeur_.transform(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.encodeur_.inverse_transform(self.reseau_.predict(self._preparer(X)))


def eegnet(tache: str, appareil: str = "cpu", n_epoques: int = N_EPOQUES) -> ClassifieurEEGNet:
    """Fabrique au même format que celles de src/pipelines.py (la tâche n'est pas utilisée)."""
    return ClassifieurEEGNet(n_epoques=n_epoques, appareil=appareil)


def matrice_alignement_euclidien(X_reference: np.ndarray) -> np.ndarray:
    """R^(-1/2), où R est la moyenne arithmétique des covariances des essais de référence.

    Alignement euclidien (He et Wu, 2019) : l'équivalent, directement sur les signaux, du
    recentrage riemannien de la phase 4. Après X -> R^(-1/2) X, la covariance moyenne du
    groupe vaut l'identité. Aucune étiquette n'est utilisée.
    """
    covariances = np.einsum("ict,idt->icd", X_reference, X_reference) / X_reference.shape[2]
    R = covariances.mean(axis=0)
    valeurs, vecteurs = np.linalg.eigh(R)
    return vecteurs @ np.diag(valeurs ** -0.5) @ vecteurs.T


def aligner(X: np.ndarray, X_reference: np.ndarray) -> np.ndarray:
    """Applique l'alignement euclidien calculé sur X_reference aux essais X."""
    return np.einsum("cd,idt->ict", matrice_alignement_euclidien(X_reference), X)
