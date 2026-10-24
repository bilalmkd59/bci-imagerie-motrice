"""Boucle d'expérience commune : un sujet, une tâche, plusieurs méthodes, mêmes plis.

Les données d'un sujet sont chargées une seule fois, puis chaque méthode est évaluée
sur exactement les mêmes essais et les mêmes plis (graine fixe), ce qui rend les
comparaisons entre méthodes appariées.
"""

from functools import partial

import pandas as pd

from src.chargement import charger_epoques
from src.evaluation import evaluer_inter_session, evaluer_intra_session

SUJETS = list(range(1, 10))
TACHES = ["binaire", "4classes"]


def evaluer_sujet(sujet: int, tache: str, methodes: dict, avec_hasard: bool = True,
                  evaluations: tuple = ("intra", "inter")) -> tuple[list, list]:
    """Évalue chaque méthode en intra-session (chaque session) puis en inter-session.

    `methodes` associe un nom à une fabrique de pipeline prenant la tâche en argument.
    `evaluations` permet de ne faire que l'une des deux (utile pour les méthodes lentes).
    Renvoie (lignes de scores, tableaux de prédictions essai par essai).
    """
    X, y, meta = charger_epoques(sujet, tache)
    session = meta["session"].to_numpy()
    lignes, predictions = [], []

    for nom_methode, fabrique_tache in methodes.items():
        fabrique = partial(fabrique_tache, tache)

        def enregistrer(evaluation, nom_session, resultat, y_vrai, y_pred):
            lignes.append({"methode": nom_methode, "sujet": sujet, "tache": tache,
                           "evaluation": evaluation, "session": nom_session,
                           "n_essais_test": len(y_vrai), **resultat})
            predictions.append(pd.DataFrame({
                "methode": nom_methode, "sujet": sujet, "tache": tache,
                "evaluation": evaluation, "session": nom_session,
                "y_vrai": y_vrai, "y_pred": y_pred}))

        for nom_session in (["0train", "1test"] if "intra" in evaluations else []):
            masque = session == nom_session
            resultat, y_pred = evaluer_intra_session(fabrique, X[masque], y[masque],
                                                     avec_hasard=avec_hasard)
            enregistrer("intra", nom_session, resultat, y[masque], y_pred)

        if "inter" in evaluations:
            app, test = session == "0train", session == "1test"
            resultat, y_pred = evaluer_inter_session(fabrique, X[app], y[app], X[test], y[test],
                                                     avec_hasard=avec_hasard)
            enregistrer("inter", "0train->1test", resultat, y[test], y_pred)
    return lignes, predictions


def evaluer_tous(methodes: dict, avec_hasard: bool = True,
                 evaluations: tuple = ("intra", "inter")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Évalue toutes les méthodes sur les 9 sujets et les 2 tâches, en affichant la progression."""
    lignes, predictions = [], []
    for tache in TACHES:
        for sujet in SUJETS:
            l, p = evaluer_sujet(sujet, tache, methodes, avec_hasard=avec_hasard,
                                 evaluations=evaluations)
            lignes += l
            predictions += p
            for nom in methodes:
                resume = ", ".join(f"{x['evaluation']} {x['session']} : acc {x['accuracy']:.3f} "
                                   f"kappa {x['kappa']:.3f}" for x in l if x["methode"] == nom)
                print(f"{tache} sujet {sujet} {nom} | {resume}", flush=True)
    return pd.DataFrame(lignes), pd.concat(predictions, ignore_index=True)


def resumer(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Un score par sujet (intra = moyenne des deux sessions), puis moyenne ± écart-type.

    Un sujet n'est compté « au-dessus du hasard » en intra-session que si ses deux sessions
    le sont (p-valeur maximale des deux), ce qui est volontairement conservateur.
    """
    agregations = {"accuracy": "mean", "kappa": "mean"}
    if "p_valeur" in scores:
        agregations |= {"hasard_accuracy_p95": "mean", "p_valeur": "max"}
    par_sujet = (scores.groupby(["methode", "tache", "evaluation", "sujet"], as_index=False)
                 .agg(agregations))
    colonnes = dict(
        accuracy_moyenne=("accuracy", "mean"), accuracy_ecart_type=("accuracy", "std"),
        kappa_moyen=("kappa", "mean"), kappa_ecart_type=("kappa", "std"),
        kappa_min=("kappa", "min"), kappa_max=("kappa", "max"),
    )
    if "p_valeur" in par_sujet:
        par_sujet["au_dessus_du_hasard"] = par_sujet["p_valeur"] < 0.05
        colonnes["sujets_au_dessus_du_hasard"] = ("au_dessus_du_hasard", "sum")
    resume = par_sujet.groupby(["methode", "tache", "evaluation"]).agg(**colonnes).reset_index()
    return par_sujet, resume
