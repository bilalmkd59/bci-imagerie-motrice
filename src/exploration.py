"""Exploration neurophysiologique : vérification des marqueurs, cartes temps-fréquence et ERD.

Convention de temps de tout ce module : t = 0 correspond à l'apparition de la flèche
(l'indice). Dans le temps d'essai officiel de la compétition, l'indice est à t = 2 s,
la croix de fixation à t = 0 s et l'imagerie dure jusqu'à t = 6 s, soit [0, 4] s ici.

Cette exploration est purement descriptive (aucun modèle n'est appris) et n'utilise
que la session d'entraînement, pour que la session de test reste vierge.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from moabb.paradigms import MotorImagery
from scipy.io import loadmat

from src.chargement import charger_brut, charger_dataset

SESSION_EXPLORATION = "0train"
CANAUX_MOTEURS = ["C3", "Cz", "C4"]
BANDES = {"mu": (8, 12), "beta": (13, 30)}

# Fenêtres en secondes relatives à l'indice
TMIN_EPOQUE, TMAX_EPOQUE = -2.5, 5.0      # marge pour les effets de bord des ondelettes
TMIN_AFFICHAGE, TMAX_AFFICHAGE = -1.5, 4.5
FENETRE_REFERENCE = (-1.5, -0.5)          # repos pendant la croix de fixation
FENETRE_ERD = (0.5, 3.0)                  # cœur de l'imagerie, après la réponse visuelle à la flèche

FREQUENCES = np.arange(4, 36, 1.0)
N_CYCLES = FREQUENCES / 2.0               # fenêtre d'environ 0,5 s pour toutes les fréquences
DECIMATION = 5                            # 250 Hz vers 50 Hz après la convolution


def _chemin_mat(sujet: int, suffixe: str = "T") -> Path:
    """Trouve le fichier .mat original de la compétition dans le cache MNE."""
    racine = Path(mne.get_config("MNE_DATA", str(Path.home() / "mne_data")))
    candidats = sorted(racine.rglob(f"A{sujet:02d}{suffixe}.mat"))
    if not candidats:
        raise FileNotFoundError(f"A{sujet:02d}{suffixe}.mat introuvable sous {racine}")
    return candidats[0]


def verifier_marqueurs(sujet: int) -> dict:
    """Compare les marqueurs MOABB au fichier .mat officiel et localise les époques du paradigme.

    Vérifie trois choses sur la session d'entraînement :
    1. le décalage entre chaque marqueur MOABB et le début d'essai officiel (champ `trial`) ;
    2. que les étiquettes de classe sont identiques ;
    3. où commence réellement une époque produite par le paradigme MotorImagery,
       en corrélant son signal avec le signal brut au marqueur (+0 s) et à +2 s.
    """
    dataset = charger_dataset()
    runs_moabb = charger_brut(sujet)[SESSION_EXPLORATION]  # télécharge aussi le .mat si besoin
    runs_mat = [r for r in loadmat(_chemin_mat(sujet), squeeze_me=True,
                                   struct_as_record=False)["data"] if np.size(r.trial) > 0]
    classe_par_code = {v: k for k, v in dataset.event_id.items()}

    decalages, etiquettes_ok = [], True
    for run_mat, raw in zip(runs_mat, runs_moabb.values()):
        debut_officiel = (run_mat.trial - 1) / run_mat.fs   # le champ est indexé à partir de 1
        decalages.append(raw.annotations.onset - debut_officiel)
        etiquettes_mat = [classe_par_code[c] for c in run_mat.y]
        etiquettes_ok &= etiquettes_mat == list(raw.annotations.description)
    decalages = np.concatenate(decalages)

    # Signal d'une époque produite par le paradigme, avec un filtre large pour comparer au brut
    paradigme = MotorImagery(n_classes=4, fmin=0.5, fmax=100)
    epoques, _, _ = paradigme.get_data(dataset, [sujet], return_epochs=True)
    signal_epoque = epoques.get_data(picks="C3")[0, 0]
    raw = next(iter(runs_moabb.values())).copy().filter(0.5, 100, picks="eeg", verbose=False)
    signal_brut = raw.get_data(picks="C3")[0]
    sfreq = raw.info["sfreq"]
    correlations = {}
    for decalage in (0, 2):
        debut = int(round((raw.annotations.onset[0] + decalage) * sfreq))
        segment = slice(50, 900)  # on ignore les bords, sensibles au filtrage
        correlations[decalage] = np.corrcoef(
            signal_epoque[segment], signal_brut[debut + segment.start:debut + segment.stop]
        )[0, 1]

    return {
        "sujet": sujet,
        "n_essais": len(decalages),
        "decalage_min_s": round(float(decalages.min()), 4),
        "decalage_max_s": round(float(decalages.max()), 4),
        "etiquettes_identiques": bool(etiquettes_ok),
        "duree_annotation_s": float(np.unique(runs_moabb["0"].annotations.duration)[0]),
        "correlation_epoque_marqueur_plus_0s": round(float(correlations[0]), 4),
        "correlation_epoque_marqueur_plus_2s": round(float(correlations[2]), 4),
        "fenetre_paradigme_moabb": str(dataset.interval),
    }


def epoquer_session(sujet: int, session: str = SESSION_EXPLORATION,
                    laplacien: bool = False) -> mne.Epochs:
    """Découpe tous les essais d'une session autour de l'indice, sur les 22 canaux EEG.

    Avec `laplacien=True`, applique un laplacien de surface (densité de courant de surface,
    splines sphériques) : chaque électrode est débarrassée de l'activité commune à ses
    voisines, ce qui limite la diffusion du rythme alpha occipital vers C3 et C4.
    """
    dataset = charger_dataset()
    liste = []
    for raw in charger_brut(sujet)[session].values():
        evenements, _ = mne.events_from_annotations(raw, event_id=dataset.event_id, verbose=False)
        liste.append(mne.Epochs(
            raw, evenements, event_id=dataset.event_id, tmin=TMIN_EPOQUE, tmax=TMAX_EPOQUE,
            picks="eeg", baseline=None, preload=True, verbose=False,
        ))
    epoques = mne.concatenate_epochs(liste, verbose=False)
    if laplacien:
        epoques = mne.preprocessing.compute_current_source_density(epoques, verbose=False)
    return epoques


def erd_temps_frequence(epoques: mne.Epochs) -> dict:
    """Calcule la carte temps-fréquence de l'ERD/ERS (en %) pour chaque classe.

    La puissance est d'abord moyennée sur les essais d'une classe, puis exprimée
    en variation relative par rapport à la période de référence :
    ERD% = (P - R) / R * 100. Valeur négative = désynchronisation (ERD).
    """
    cartes = {}
    for classe in epoques.event_id:
        tfr = epoques[classe].compute_tfr(
            method="morlet", freqs=FREQUENCES, n_cycles=N_CYCLES, average=True,
            return_itc=False, decim=DECIMATION, verbose=False,
        )
        tfr.apply_baseline(FENETRE_REFERENCE, mode="percent", verbose=False)
        tfr.crop(TMIN_AFFICHAGE, TMAX_AFFICHAGE)
        tfr.data *= 100  # le mode "percent" de MNE renvoie un ratio, on le passe en %
        cartes[classe] = tfr
    return cartes


def tableau_erd(cartes: dict, sujet: int) -> pd.DataFrame:
    """Moyenne l'ERD% par canal, classe et bande dans la fenêtre d'imagerie."""
    lignes = []
    for classe, tfr in cartes.items():
        masque_t = (tfr.times >= FENETRE_ERD[0]) & (tfr.times <= FENETRE_ERD[1])
        for bande, (fmin, fmax) in BANDES.items():
            masque_f = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
            valeurs = tfr.data[:, masque_f][:, :, masque_t].mean(axis=(1, 2))
            for canal, valeur in zip(tfr.ch_names, valeurs):
                lignes.append({"sujet": sujet, "classe": classe, "bande": bande,
                               "canal": canal, "erd_pct": valeur})
    return pd.DataFrame(lignes)


def decours_temporel(cartes: dict, bande: str) -> dict:
    """Renvoie, pour chaque classe, l'ERD% moyenné sur une bande : tableau (canaux, temps)."""
    fmin, fmax = BANDES[bande]
    decours = {}
    for classe, tfr in cartes.items():
        masque_f = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
        decours[classe] = tfr.data[:, masque_f].mean(axis=1)
    return decours


def indices_lateralisation(erd: pd.DataFrame) -> pd.DataFrame:
    """Indice de latéralisation par sujet, bande et main imaginée.

    Indice = ERD% controlatéral - ERD% ipsilatéral (en points de pourcentage).
    Main gauche : C4 - C3. Main droite : C3 - C4.
    Valeur négative = désynchronisation plus forte du côté opposé à la main, comme attendu.
    """
    controlateral = {"left_hand": ("C4", "C3"), "right_hand": ("C3", "C4")}
    pivot = erd[erd["canal"].isin(["C3", "C4"])].pivot_table(
        index=["sujet", "bande", "classe"], columns="canal", values="erd_pct"
    )
    lignes = []
    for (sujet, bande, classe), ligne in pivot.iterrows():
        if classe not in controlateral:
            continue
        contra, ipsi = controlateral[classe]
        indice = ligne[contra] - ligne[ipsi]
        lignes.append({"sujet": sujet, "bande": bande, "classe": classe,
                       "erd_controlateral_pct": ligne[contra], "erd_ipsilateral_pct": ligne[ipsi],
                       "indice_lateralisation": indice, "controlateral_attendu": indice < 0})
    return pd.DataFrame(lignes)
