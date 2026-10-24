"""Protocoles d'évaluation communs à toutes les méthodes.

- intra-session : validation croisée stratifiée en 5 plis, dans une même session ;
- inter-session : apprentissage sur la session 1 (0train), test sur la session 2 (1test),
  comme dans la compétition originale ;
- niveau du hasard : on refait exactement la même évaluation avec des étiquettes permutées.

Les plis sont tirés avec une graine fixe : deux méthodes évaluées sur les mêmes données
voient exactement les mêmes plis, ce qui permet des comparaisons appariées.
"""

from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from src.chargement import GRAINE

N_PLIS = 5
N_PERMUTATIONS = 100


def scores(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy et kappa de Cohen (métrique officielle de la compétition)."""
    return {"accuracy": accuracy_score(y_vrai, y_pred),
            "kappa": cohen_kappa_score(y_vrai, y_pred)}


def predire_validation_croisee(fabrique: Callable, X: np.ndarray, y: np.ndarray,
                               graine: int = GRAINE) -> np.ndarray:
    """Prédit chaque essai par un modèle qui ne l'a jamais vu (validation croisée stratifiée).

    Un pipeline neuf est créé et appris à chaque pli : CSP et classifieur ne voient
    jamais les essais du pli de test.
    """
    plis = StratifiedKFold(n_splits=N_PLIS, shuffle=True, random_state=graine)
    y_pred = np.empty_like(y)
    for idx_app, idx_test in plis.split(X, y):
        modele = fabrique().fit(X[idx_app], y[idx_app])
        y_pred[idx_test] = modele.predict(X[idx_test])
    return y_pred


def predire_apprentissage_test(fabrique: Callable, X_app: np.ndarray, y_app: np.ndarray,
                               X_test: np.ndarray) -> np.ndarray:
    """Apprend sur un ensemble et prédit un autre (ici : session 1 vers session 2)."""
    return fabrique().fit(X_app, y_app).predict(X_test)


def _permutation_intra(y_perm: np.ndarray, fabrique: Callable, X: np.ndarray) -> dict:
    """Validation croisée complète avec étiquettes permutées (elles deviennent la « vérité »)."""
    return scores(y_perm, predire_validation_croisee(fabrique, X, y_perm))


def _permutation_inter(y_perm: np.ndarray, fabrique: Callable, X_app: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Apprentissage avec étiquettes permutées, test sur les vraies étiquettes de la session 2."""
    return scores(y_test, predire_apprentissage_test(fabrique, X_app, y_perm, X_test))


def niveau_hasard(evaluer: Callable, y: np.ndarray, score_observe: float, donnees: dict,
                  n_permutations: int = N_PERMUTATIONS, graine: int = GRAINE) -> dict:
    """Estime le niveau du hasard par permutation des étiquettes.

    `evaluer(y_permute, **donnees)` doit refaire toute l'évaluation (apprentissage compris)
    avec les étiquettes permutées et renvoyer un dictionnaire de scores. La distribution
    obtenue est celle d'un modèle qui n'a aucune information réelle sur les classes.
    La p-valeur est (nombre de permutations au moins aussi bonnes + 1) / (n + 1).

    Les données passent en arguments explicites (et non dans une fonction anonyme) :
    joblib peut alors les partager entre processus par projection mémoire au lieu de
    les recopier pour chaque permutation.
    """
    generateur = np.random.default_rng(graine)
    permutations = [generateur.permutation(y) for _ in range(n_permutations)]
    resultats = Parallel(n_jobs=-1)(delayed(evaluer)(y_perm, **donnees)
                                    for y_perm in permutations)
    acc = np.array([r["accuracy"] for r in resultats])
    kappa = np.array([r["kappa"] for r in resultats])
    return {
        "hasard_accuracy_moyenne": acc.mean(),
        "hasard_accuracy_p95": np.percentile(acc, 95),
        "hasard_kappa_moyen": kappa.mean(),
        "p_valeur": (np.sum(acc >= score_observe) + 1) / (n_permutations + 1),
    }


def evaluer_intra_session(fabrique: Callable, X: np.ndarray, y: np.ndarray,
                          avec_hasard: bool = True) -> tuple[dict, np.ndarray]:
    """Validation croisée stratifiée dans une session. Renvoie (scores, prédictions)."""
    y_pred = predire_validation_croisee(fabrique, X, y)
    resultat = scores(y, y_pred)
    if avec_hasard:
        # En validation croisée, les étiquettes permutées deviennent la « vérité » de l'expérience
        resultat |= niveau_hasard(_permutation_intra, y, resultat["accuracy"],
                                  {"fabrique": fabrique, "X": X})
    return resultat, y_pred


def evaluer_inter_session(fabrique: Callable, X_app: np.ndarray, y_app: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          avec_hasard: bool = True) -> tuple[dict, np.ndarray]:
    """Apprentissage sur la session 1, test sur la session 2. Renvoie (scores, prédictions)."""
    y_pred = predire_apprentissage_test(fabrique, X_app, y_app, X_test)
    resultat = scores(y_test, y_pred)
    if avec_hasard:
        # On permute les étiquettes d'apprentissage ; le test garde ses vraies étiquettes
        resultat |= niveau_hasard(_permutation_inter, y_app, resultat["accuracy"],
                                  {"fabrique": fabrique, "X_app": X_app, "X_test": X_test,
                                   "y_test": y_test})
    return resultat, y_pred
