"""Phase 0 : vérifie que l'environnement fonctionne et que le sujet 1 se charge.

Usage (depuis la racine du projet) :
    .venv\\Scripts\\python.exe -m src.verifier_environnement

Le résumé est sauvegardé dans results/phase0_verification_sujet1.csv.
"""

from pathlib import Path

import mne
import moabb
import numpy as np
import pandas as pd
import pyriemann
import sklearn

from src.chargement import charger_brut, charger_dataset, fixer_graine

RACINE = Path(__file__).resolve().parents[1]


def main() -> None:
    fixer_graine()
    mne.set_log_level("WARNING")

    print("Versions :")
    for nom, module in [("mne", mne), ("moabb", moabb), ("pyriemann", pyriemann),
                        ("scikit-learn", sklearn), ("numpy", np), ("pandas", pd)]:
        print(f"  {nom:<13}{module.__version__}")

    dataset = charger_dataset()
    print(f"\nSujets : {dataset.subject_list}")
    print(f"Classes : {dataset.event_id}")
    print(f"Intervalle d'imagerie déclaré par MOABB (s depuis le début de l'essai) : {dataset.interval}")

    sessions = charger_brut(1)
    lignes = []
    for session, runs in sessions.items():
        for run, raw in runs.items():
            evenements, _ = mne.events_from_annotations(raw, verbose=False)
            types = raw.get_channel_types()
            lignes.append({
                "session": session,
                "run": run,
                "n_eeg": types.count("eeg"),
                "n_eog": types.count("eog"),
                "n_stim": types.count("stim"),
                "frequence_hz": raw.info["sfreq"],
                "duree_s": round(raw.times[-1], 1),
                "n_evenements": len(evenements),
            })

    resume = pd.DataFrame(lignes)
    print("\nSujet 1, structure des enregistrements :")
    print(resume.to_string(index=False))
    print("\nNombre d'essais par session :")
    print(resume.groupby("session")["n_evenements"].sum().to_string())

    sortie = RACINE / "results" / "phase0_verification_sujet1.csv"
    resume.to_csv(sortie, index=False)
    print(f"\nRésumé sauvegardé dans {sortie.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
