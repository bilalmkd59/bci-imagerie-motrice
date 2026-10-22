"""Style commun des figures du projet (PNG en 150 dpi dans reports/figures/)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # rendu sans fenêtre, pour les scripts
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_FIGURES = RACINE / "reports" / "figures"
DPI = 150

# Couleurs fixes par classe : une classe garde toujours la même couleur dans tout le projet.
COULEURS_CLASSES = {
    "left_hand": "#2a78d6",   # bleu
    "right_hand": "#eb6834",  # orange
    "feet": "#1baf7a",        # vert d'eau
    "tongue": "#eda100",      # jaune
}
NOMS_CLASSES = {
    "left_hand": "Main gauche",
    "right_hand": "Main droite",
    "feet": "Pieds",
    "tongue": "Langue",
}

ENCRE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"

# Carte divergente pour l'ERD/ERS : bleu = baisse de puissance (désynchronisation),
# rouge = hausse (synchronisation), gris neutre = pas de changement.
CARTE_ERD = LinearSegmentedColormap.from_list(
    "erd", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"]
)


def appliquer_style() -> None:
    """Règle l'apparence matplotlib : axes discrets, grille légère, texte sobre."""
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": ENCRE_SECONDAIRE,
        "axes.titlecolor": ENCRE,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRILLE,
        "grid.linewidth": 0.6,
        "xtick.color": ENCRE_DISCRETE,
        "ytick.color": ENCRE_DISCRETE,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
    })


def sauvegarder(fig, nom: str) -> Path:
    """Enregistre la figure en PNG 150 dpi dans reports/figures/ et la ferme."""
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    chemin = DOSSIER_FIGURES / nom
    fig.savefig(chemin, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return chemin
