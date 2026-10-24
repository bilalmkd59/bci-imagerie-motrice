"""Partie explicative du rapport de synthèse, rédigée pour un lecteur débutant.

Les explications sont écrites ici ; tous les chiffres sont lus dans les CSV de results/
au moment de la génération (aucun chiffre saisi à la main).
"""

import pandas as pd

from src.figures import RACINE

RESULTATS = RACINE / "results"


def _n(valeur: float, decimales: int = 2) -> str:
    """Nombre au format français (virgule décimale, vrai signe moins)."""
    return f"{valeur:.{decimales}f}".replace(".", ",").replace("-", "−")


def _pct(valeur: float) -> str:
    return f"{100 * valeur:.0f} %"


def _lire(nom: str) -> pd.DataFrame:
    return pd.read_csv(RESULTATS / nom)


class Chiffres:
    """Accès simple aux valeurs du tableau comparatif."""

    def __init__(self, tableau: pd.DataFrame):
        self.tableau = tableau

    def kappa(self, tache: str, protocole: str, methode: str) -> float:
        ligne = self.tableau[(self.tableau["tache"] == tache)
                             & (self.tableau["protocole"] == protocole)
                             & (self.tableau["methode"] == methode)]
        return float(ligne["kappa_moyen"].iloc[0]) if not ligne.empty else float("nan")

    def accuracy(self, tache: str, protocole: str, methode: str) -> float:
        ligne = self.tableau[(self.tableau["tache"] == tache)
                             & (self.tableau["protocole"] == protocole)
                             & (self.tableau["methode"] == methode)]
        return float(ligne["accuracy_moyenne"].iloc[0]) if not ligne.empty else float("nan")


# ---------------------------------------------------------------- introduction

def resume_express(c: Chiffres) -> str:
    perte = c.kappa("4classes", "inter", "TS+LR") - c.kappa("4classes", "loso",
                                                            "TS+LR, sans recentrage")
    gain = (c.kappa("4classes", "loso", "TS+LR, recentrage (réf. session 1)")
            - c.kappa("4classes", "loso", "TS+LR, sans recentrage"))
    return f"""\
## En bref

- **Objectif :** deviner, à partir de l'activité électrique du cerveau (EEG), quel mouvement une personne
  *imagine* (main gauche, main droite, pieds ou langue), sans qu'elle bouge.
- **Le signal existe et il est latéralisé :** imaginer la main droite fait baisser l'activité rythmique au-dessus
  du cortex moteur gauche, et inversement (phase 1).
- **Méthode de référence (CSP + LDA) :** kappa moyen de {_n(c.kappa("4classes", "inter", "CSP+LDA"))} en 4 classes
  (accuracy {_pct(c.accuracy("4classes", "inter", "CSP+LDA"))}, hasard à 25 %), dans les conditions de la compétition.
- **Meilleure méthode : la géométrie riemannienne (TS+LR)**, kappa {_n(c.kappa("4classes", "inter", "TS+LR"))}
  (accuracy {_pct(c.accuracy("4classes", "inter", "TS+LR"))}).
- **Sur une personne jamais vue,** le kappa chute de {_n(perte)} ; un recentrage mathématique sans aucune étiquette
  du nouvel utilisateur en récupère {_n(gain)}.
- **Le deep learning (EEGNet)** est en retrait avec peu de données par personne, mais rejoint les meilleures
  méthodes quand on l'entraîne sur plusieurs personnes.
- **La plus grande source de variation, ce sont les personnes elles-mêmes,** pas les algorithmes."""


CONTEXTE = """\
## 1. Le problème, expliqué simplement

### Qu'est-ce qu'une interface cerveau-machine par imagerie motrice ?

Une **interface cerveau-machine** (BCI, *brain-computer interface*) traduit l'activité du cerveau en commande
pour une machine : déplacer un curseur, piloter un fauteuil roulant, une prothèse. C'est un enjeu majeur pour les
personnes paralysées, par exemple après un AVC ou atteintes de sclérose latérale amyotrophique.

Dans l'**imagerie motrice**, l'utilisateur ne bouge pas : il *imagine* un mouvement. On enregistre son activité
cérébrale avec un **EEG** (électroencéphalogramme) : des électrodes posées sur le cuir chevelu mesurent de très
petites tensions électriques (quelques millionièmes de volt) produites par des millions de neurones.

### Quel signal cherche-t-on ?

Au repos, les neurones du cortex moteur oscillent ensemble, à la manière d'une foule qui applaudit en rythme :
cela produit des ondes régulières, le **rythme mu (8-12 Hz)** et le **rythme beta (13-30 Hz)**.
Quand on prépare, fait **ou imagine** un mouvement, ces neurones se mettent à travailler chacun de leur côté :
le rythme se brouille et sa puissance baisse. C'est la **désynchronisation liée à l'événement (ERD)**.

La motricité est **croisée** : la main droite est commandée par l'hémisphère gauche (électrode C3), la main
gauche par l'hémisphère droit (électrode C4). Imaginer la main droite doit donc faire baisser la puissance
sur C3 plus que sur C4. Ce **contraste spatial** est ce que les algorithmes apprennent à reconnaître.

### Les données

Jeu de données public **BCI Competition IV 2a** (BNCI2014_001), chargé avec la bibliothèque **MOABB** :
9 personnes, 22 électrodes EEG (plus 3 électrodes oculaires non utilisées pour la classification),
250 mesures par seconde, 4 mouvements imaginés. Chaque personne a fait **2 sessions, des jours différents**,
de 288 essais chacune (72 par mouvement). Un essai dure environ 8 secondes : une croix apparaît (t = 0 s),
une flèche indique le mouvement à imaginer (t = 2 s), la personne l'imagine jusqu'à t = 6 s."""


METHODE_EVALUATION = """\
## 2. Comment juger un algorithme honnêtement

### La règle d'or : ne jamais évaluer sur ce qui a servi à apprendre

Un algorithme qui a « vu » les réponses pendant son apprentissage peut les recracher par cœur : son score serait
excellent et totalement trompeur. Toute la démarche du projet vise à éviter cette **fuite de données** :

- les essais de test ne servent **jamais** à l'apprentissage, y compris pour les étapes intermédiaires
  (filtres spatiaux CSP, moyennes de référence) : tout ce qui apprend est dans un `Pipeline` scikit-learn
  réappris à chaque découpage ;
- les réglages (fenêtre de temps, bande de fréquences, nombre de composantes, hyperparamètres d'EEGNet) ont été
  **fixés à l'avance**, jamais ajustés pour améliorer un score de test ;
- l'exploration de la phase 1 n'utilise que la première session, pour que la seconde reste vierge.

### Trois façons de tester, de la plus facile à la plus exigeante

| Protocole | Apprentissage | Test | Question posée |
|---|---|---|---|
| **Intra-session** | 4/5 des essais d'une session | le 1/5 restant, même session | Le signal est-il décodable ? |
| **Inter-session** | toute la session 1 | toute la session 2 (un autre jour) | Le modèle tient-il d'un jour à l'autre ? (protocole de la compétition) |
| **Sujet jamais vu** | les 8 autres personnes | la session 2 de la personne testée | Peut-on se passer de calibration ? |

L'intra-session utilise une **validation croisée stratifiée en 5 plis** : on coupe les essais en 5 paquets
(avec la même proportion de chaque mouvement), on apprend sur 4 et on teste sur le 5e, cinq fois de suite.
Chaque essai est ainsi prédit une fois, par un modèle qui ne l'a jamais vu.

### Les mesures de performance

- **Accuracy** : la proportion d'essais bien classés. Simple, mais trompeuse seule : 50 % est excellent pour un choix
  parmi 100, et nul pour un choix entre 2.
- **Kappa de Cohen** : l'accuracy corrigée du hasard. **0 = pas mieux que le hasard, 1 = parfait.**
  Formule : kappa = (accuracy observée − accuracy due au hasard) / (1 − accuracy due au hasard).
  C'est la mesure officielle de la compétition ; elle permet de comparer 2 classes et 4 classes sur la même échelle.
- **Matrice de confusion** : un tableau qui indique, pour chaque mouvement réel, ce que l'algorithme a prédit.
  Elle montre *quelles* classes sont confondues.

### Les tests statistiques utilisés

- **Test par permutation (niveau du hasard).** On mélange au hasard les étiquettes (on « ment » à l'algorithme
  sur ce que la personne imaginait), puis on refait tout l'apprentissage et l'évaluation ; on recommence 100 fois.
  On obtient ainsi les scores qu'un algorithme peut atteindre **sans aucune information réelle**. Si le vrai score
  dépasse 95 % de ces scores « au hasard », il est significatif (p < 0,05). C'est plus fiable que de supposer
  25 % ou 50 % : avec seulement 144 essais de test, le hasard peut donner jusqu'à 60 % ou plus en binaire.
- **Test de Wilcoxon apparié (comparer deux méthodes).** On compare deux méthodes **sur les mêmes personnes**,
  avec les mêmes essais et les mêmes découpages : pour chaque personne, on regarde laquelle gagne et de combien.
  « Apparié » est essentiel, car les écarts entre personnes sont bien plus grands que les écarts entre méthodes.
  Ce test ne suppose pas que les données suivent une loi normale, ce qu'on ne peut pas vérifier avec 9 personnes.
  Avec 9 personnes, la plus petite p-valeur possible est d'environ 0,004 (cas où les 9 vont dans le même sens).
- **Correction de Holm (comparaisons multiples).** Faire beaucoup de tests augmente le risque qu'un d'entre eux soit
  « significatif » par pur hasard (comme lancer une pièce 20 fois finit par donner une série surprenante). La
  correction de Holm rend chaque test plus exigeant pour que le risque *global* reste à 5 %. Les p-valeurs sont donc
  données avant (« p ») et après correction (« p corrigée »)."""


# ---------------------------------------------------------------- phases

def phase0() -> str:
    return """\
### Phase 0 : préparer un environnement reproductible

- **But :** que n'importe qui puisse relancer le projet et obtenir les mêmes chiffres.
- **Démarche :** environnement Python isolé (`.venv`), versions exactes des bibliothèques figées dans
  `requirements.txt`, graine aléatoire fixée à 42, dépôt git, test de chargement d'une première personne.
- **Réflexion :** les bibliothèques d'EEG (MOABB, braindecode) changent souvent d'interface d'une version à l'autre ;
  les noms de fonctions ont donc été vérifiés dans la version installée plutôt que supposés.
- **À retenir :** sans reproductibilité, un résultat n'a pas de valeur scientifique."""


def phase1() -> str:
    verif = _lire("phase1_verification_marqueurs.csv")
    erd = _lire("phase1_erd_par_sujet.csv")
    lat = _lire("phase1_lateralisation.csv")
    erd_lap = (erd[(erd["derivation"] == "laplacien") & (erd["bande"] == "mu")]
               .groupby(["classe", "canal"])["erd_pct"].mean())
    part = lat.groupby("derivation")["controlateral_attendu"].mean()
    return f"""\
### Phase 1 : vérifier que le signal recherché existe

- **But :** avant tout apprentissage automatique, s'assurer que l'imagerie motrice laisse une trace visible et
  conforme à la neurophysiologie (l'ERD latéralisée).
- **Démarche :**
  1. *Vérifier le découpage des essais.* Les marqueurs de MOABB ont été comparés aux fichiers officiels de la
     compétition : décalage constant de {_n(verif["decalage_min_s"].min(), 3)} s (le marqueur est placé sur la
     flèche, pas au début de l'essai), étiquettes identiques pour les 9 personnes, et les essais découpés par MOABB
     coïncident avec le signal brut (corrélation d'au moins {_n(verif["correlation_epoque_marqueur_plus_0s"].min())}).
     La fenêtre analysée couvre donc bien toute la période d'imagerie. Une erreur de 2 secondes ici aurait faussé tout
     le projet sans aucun message d'erreur.
  2. *Cartes temps-fréquence.* Pour chaque instant et chaque fréquence, on calcule la variation de puissance par
     rapport au repos (avant la flèche) : ERD % = (puissance − repos) / repos × 100. Une valeur négative est une ERD.
  3. *Filtre spatial laplacien.* Le crâne diffuse les signaux : chaque électrode capte un mélange de sources, dont le
     puissant rythme alpha visuel de l'arrière de la tête. Le laplacien soustrait à chaque électrode l'activité de ses
     voisines pour ne garder que la source située juste dessous.
- **Résultats (moyenne des 9 personnes, rythme mu, laplacien) :** en imaginant la **main droite**, la puissance baisse de
  {_n(-erd_lap[("right_hand", "C3")], 1)} % sur C3 (hémisphère gauche) contre {_n(-erd_lap[("right_hand", "C4")], 1)} % sur C4 ;
  en imaginant la **main gauche**, de {_n(-erd_lap[("left_hand", "C4")], 1)} % sur C4 contre {_n(-erd_lap[("left_hand", "C3")], 1)} % sur C3.
  La désynchronisation est plus forte du côté opposé à la main, nettement pour la main droite, plus modestement
  pour la main gauche. Ce schéma se retrouve dans
  {_pct(part["laplacien"])} des cas (personne × bande × main) avec le laplacien, contre {_pct(part["ref_commune"])} sans.
- **À retenir :** le signal existe et il est latéralisé comme le prévoit la physiologie, mais avec de grandes différences
  entre personnes. Figures : `reports/figures/phase1_*.png`."""


def phase2(c: Chiffres) -> str:
    resume = _lire("phase2_resume.csv").set_index(["tache", "evaluation"])
    return f"""\
### Phase 2 : la méthode de référence, CSP + LDA

- **But :** établir un point de comparaison avec la méthode classique du domaine.
- **Démarche :**
  - *Filtrage 8-30 Hz* pour ne garder que les rythmes mu et beta.
  - *CSP (Common Spatial Patterns)* : l'algorithme cherche des combinaisons des 22 électrodes dont la puissance est
    maximale pour une classe et minimale pour l'autre. Il apprend en quelque sorte où regarder sur la tête.
    Les cartes de ses **motifs** (`phase2_csp_*.png`) montrent, chez les bonnes personnes, des foyers sur C3 et C4,
    exactement là où la phase 1 trouvait l'ERD.
  - *LDA (analyse discriminante linéaire)* : un classifieur simple qui trace une frontière entre les classes à partir
    de ces puissances.
  - Évaluation en binaire (main gauche contre main droite) puis en 4 classes, en intra- et inter-session,
    avec 100 permutations par évaluation pour mesurer le niveau du hasard.
- **Résultats :** en 4 classes inter-session, kappa {_n(c.kappa("4classes", "inter", "CSP+LDA"))}, toutes les personnes
  au-dessus du hasard ({int(resume.loc[("4classes", "inter"), "sujets_au_dessus_du_hasard"])}/9).
  En binaire inter-session, kappa {_n(c.kappa("binaire", "inter", "CSP+LDA"))},
  {int(resume.loc[("binaire", "inter"), "sujets_au_dessus_du_hasard"])}/9 personnes au-dessus du hasard.
  Ces valeurs sont dans la fourchette publiée pour cette méthode sur ce jeu de données, ce qui écarte une fuite de données.
- **À retenir :** une méthode simple et interprétable fonctionne bien chez certaines personnes et presque pas chez
  d'autres (« illettrisme BCI »). Le passage d'un jour à l'autre dégrade les résultats en 4 classes : c'est la
  **non-stationnarité** du signal (casque replacé, fatigue, impédance des électrodes)."""


def phase3(c: Chiffres) -> str:
    w = _lire("phase3_wilcoxon.csv")
    r = w[(w["tache"] == "4classes") & (w["evaluation"] == "inter")
          & (w["methode_a"] == "TS+LR") & (w["methode_b"] == "CSP+LDA")].iloc[0]
    return f"""\
### Phase 3 : la géométrie riemannienne

- **But :** tester une famille de méthodes plus récente, réputée plus performante et plus robuste.
- **Idée :** chaque essai est résumé par sa **matrice de covariance** (22 × 22) : la puissance de chaque électrode et
  la façon dont les électrodes varient ensemble. Ces matrices ont une contrainte (elles sont « définies positives ») qui
  les place sur une surface courbe, une **variété**, et non dans un espace plat. Calculer une moyenne « en ligne
  droite » entre deux matrices les déforme (effet de gonflement, figure `phase3_intuition_variete.png`) ;
  la géométrie riemannienne mesure les distances *le long* de la surface.
- **Deux méthodes :**
  - *MDM* : on calcule une matrice moyenne par mouvement, et chaque nouvel essai est attribué à la moyenne la plus proche.
  - *TS+LR* : on « déplie » localement la surface, comme une carte plane autour d'une ville (l'**espace tangent**),
    puis on applique une régression logistique.
- **Résultats (4 classes, inter-session) :** TS+LR obtient un kappa de {_n(c.kappa("4classes", "inter", "TS+LR"))}, contre
  {_n(c.kappa("4classes", "inter", "CSP+LDA"))} pour CSP+LDA et {_n(c.kappa("4classes", "inter", "MDM"))} pour MDM.
  TS+LR fait mieux que CSP+LDA chez {int(r["sujets_a_meilleur"])} personnes sur 9 (p = {_n(r["p_valeur"], 3)}),
  mais après correction pour les 12 tests réalisés, la différence n'est plus significative
  (p corrigée = {_n(r["p_valeur_holm"], 2)}).
- **À retenir :** TS+LR a la meilleure moyenne partout, sans aucun réglage. Mais avec 9 personnes, cet avantage reste
  une tendance et non une preuve : il faudrait davantage de personnes pour conclure."""


def phase4(c: Chiffres) -> str:
    rec = _lire("phase4_recuperation.csv").set_index(["tache", "alignement"])
    propre = c.kappa("4classes", "inter", "TS+LR")
    sans = c.kappa("4classes", "loso", "TS+LR, sans recentrage")
    calib = c.kappa("4classes", "loso", "TS+LR, recentrage (réf. session 1)")
    transd = c.kappa("4classes", "loso", "TS+LR, recentrage transductif")
    return f"""\
### Phase 4 : une personne jamais vue

- **But :** mesurer si un modèle appris sur d'autres personnes fonctionne sur un nouvel utilisateur, sans séance de
  calibration étiquetée. C'est la question pratique la plus importante pour une BCI « prête à l'emploi ».
- **Démarche :** *leave-one-subject-out* : pour chaque personne, on apprend sur les 8 autres et on teste sur elle.
  On compare au modèle propre à cette personne (phase 3), sur exactement les mêmes essais de test.
- **Le problème :** chaque personne occupe sa propre région de la variété (anatomie, position du casque). Dans la figure
  `phase4_espace_tangent.png`, les essais se regroupent d'abord par personne, pas par mouvement imaginé.
- **La solution testée : le recentrage riemannien.** Pour chaque personne, on calcule sa matrice moyenne et on ramène
  toutes ses matrices autour d'un point commun (l'identité). On ne garde ainsi que les variations *par rapport à son
  état habituel*, qui se ressemblent davantage d'une personne à l'autre. **Aucune étiquette n'est utilisée.** Deux variantes :
  référence calculée sur la **session 1** de la nouvelle personne (réaliste : aucune donnée de test utilisée) ou sur
  ses **essais de test** eux-mêmes (« transductive », plus optimiste).
- **Résultats (TS+LR, 4 classes) :** modèle propre à la personne {_n(propre)} ; modèle des autres sans recentrage {_n(sans)} ;
  avec recentrage {_n(calib)} (référence session 1) et {_n(transd)} (transductif). Le recentrage récupère
  {_pct(rec.loc[("4classes", "calibration"), "part_de_la_perte_recuperee"])} de la perte en conditions réalistes et
  {_pct(rec.loc[("4classes", "transductif"), "part_de_la_perte_recuperee"])} en transductif.
- **À retenir :** un modèle générique perd beaucoup sur un nouvel utilisateur ; le recentrage en rattrape une part
  importante sans aucune étiquette, mais ne remplace pas une calibration. Il pourrait en revanche la raccourcir."""


def phase5(c: Chiffres) -> str:
    diag = _lire("phase5_diagnostic.csv")
    s7 = diag[diag["sujet"] == 7]
    alignement = _lire("phase5_alignement_loso.csv").set_index("tache")
    return f"""\
### Phase 5 : le deep learning (EEGNet)

- **But :** comparer un réseau de neurones conçu pour l'EEG aux méthodes précédentes, dans les mêmes conditions.
- **Démarche :** EEGNet (bibliothèque braindecode) apprend lui-même ses filtres temporels et spatiaux. Pour une
  comparaison équitable, il reçoit exactement les mêmes essais et passe par la même boucle d'évaluation ; ses réglages
  sont ceux de l'article original, fixés à l'avance, sans arrêt précoce (il faudrait des données de validation, et les
  prendre sur le test serait une fuite). Les protocoles les plus lourds ont été calculés sur GPU (Google Colab).
- **Résultats :** avec peu d'essais par personne, EEGNet est en retrait (binaire intra-session : kappa
  {_n(c.kappa("binaire", "intra", "EEGNet"))} contre {_n(c.kappa("binaire", "intra", "CSP+LDA"))} pour CSP+LDA).
  Le diagnostic l'explique : sur la personne 7, il atteint un kappa de {_n(s7["kappa_apprentissage"].min())} à
  {_n(s7["kappa_apprentissage"].max())} sur les données d'apprentissage mais de {_n(s7["kappa_test"].min())} à
  {_n(s7["kappa_test"].max())} en test : il **apprend par cœur** (surapprentissage). Entraîné sur 8 personnes (protocole
  « sujet jamais vu », 4 classes), il obtient {_n(c.kappa("4classes", "loso", "EEGNet, alignement (réf. session 1)"))},
  au niveau de TS+LR recentré ({_n(c.kappa("4classes", "loso", "TS+LR, recentrage (réf. session 1)"))}).
  L'alignement des signaux l'aide en 4 classes (+{_n(alignement.loc["4classes", "difference_moyenne"])} de kappa,
  {int(alignement.loc["4classes", "sujets_ameliores"])}/9 personnes), sans que ce soit significatif.
- **À retenir :** le deep learning a besoin de beaucoup de données. Avec quelques centaines d'essais d'une personne,
  les méthodes qui intègrent la connaissance du domaine (la puissance des rythmes mu et beta) sont meilleures et plus
  stables ; avec plusieurs milliers d'essais de plusieurs personnes, EEGNet rejoint les meilleures."""


def phase6() -> str:
    return """\
### Phase 6 : synthèse et reproductibilité

- **But :** pouvoir tout relancer d'une seule commande et présenter les résultats sans erreur de recopie.
- **Démarche :** le script `src/tout_relancer.py` enchaîne toutes les étapes ; ce rapport est généré par
  `src/synthese.py` directement à partir des fichiers de résultats.
- **À retenir :** chaque chiffre de ce document provient d'une exécution réelle et sauvegardée dans `results/`."""


GLOSSAIRE = """\
## Glossaire

| Terme | Signification |
|---|---|
| **Accuracy** | Proportion d'essais correctement classés. |
| **Apparié (test)** | Comparaison faite personne par personne, chaque personne étant sa propre référence. |
| **BCI** | Interface cerveau-machine (*brain-computer interface*). |
| **Covariance (matrice de)** | Tableau qui décrit la puissance de chaque électrode et la façon dont elles varient ensemble. |
| **CSP** | *Common Spatial Patterns* : filtres spatiaux qui maximisent la différence de puissance entre deux classes. |
| **EEG** | Électroencéphalogramme : enregistrement de l'activité électrique du cerveau par des électrodes sur le crâne. |
| **EEGNet** | Réseau de neurones convolutif compact conçu pour l'EEG. |
| **Époque (deep learning)** | Un passage complet sur toutes les données d'apprentissage. |
| **ERD / ERS** | Baisse (désynchronisation) / hausse (synchronisation) de la puissance d'un rythme cérébral liée à un événement. |
| **Espace tangent** | Version « dépliée » et plane de la variété autour d'un point, où l'on peut utiliser des outils classiques. |
| **Fuite de données** | Situation où une information du test influence l'apprentissage, ce qui gonfle artificiellement les scores. |
| **Illettrisme BCI** | Phénomène où une partie des utilisateurs n'arrive pas à produire un signal décodable. |
| **Kappa de Cohen** | Accuracy corrigée du hasard : 0 = hasard, 1 = parfait. |
| **Laplacien (de surface)** | Filtre spatial qui soustrait l'activité des électrodes voisines pour isoler la source locale. |
| **LDA** | Analyse discriminante linéaire : classifieur qui trace une frontière droite entre les classes. |
| **Leave-one-subject-out** | Évaluation où chaque personne est testée par un modèle appris sur toutes les autres. |
| **MDM** | *Minimum Distance to Mean* : classe chaque essai selon la moyenne riemannienne la plus proche. |
| **Non-stationnarité** | Le signal d'une même personne change d'un jour, voire d'une minute, à l'autre. |
| **p-valeur** | Probabilité d'observer un écart au moins aussi grand si, en réalité, il n'y avait aucune différence. |
| **Permutation (test par)** | Refaire l'évaluation avec des étiquettes mélangées pour mesurer ce que donne le hasard. |
| **Recentrage riemannien** | Transformation qui ramène la moyenne des matrices de chaque personne à un point commun. |
| **Rythmes mu / beta** | Oscillations du cortex moteur, vers 8-12 Hz et 13-30 Hz. |
| **Surapprentissage** | Le modèle apprend par cœur les données d'apprentissage et généralise mal. |
| **TS+LR** | Projection dans l'espace tangent puis régression logistique. |
| **Validation croisée** | Découpage répété des données en parties d'apprentissage et de test. |
| **Variété riemannienne** | Surface courbe sur laquelle vivent les matrices de covariance. |
| **Wilcoxon (test de)** | Test statistique non paramétrique pour comparer deux méthodes sur les mêmes personnes. |"""


def verifier_affirmations(c: Chiffres) -> list:
    """Vérifie sur les données les affirmations qualitatives du texte ; renvoie celles qui sont fausses.

    Le texte explicatif contient quelques phrases générales (« TS+LR a la meilleure moyenne »...).
    Si une nouvelle exécution les contredisait, elles seraient signalées en tête du rapport.
    """
    problemes = []
    configurations = [(t, p) for t in ["binaire", "4classes"] for p in ["intra", "inter"]]
    if not all(c.kappa(t, p, "TS+LR") >= max(c.kappa(t, p, m) for m in ["CSP+LDA", "MDM", "EEGNet"])
               for t, p in configurations):
        problemes.append("TS+LR n'a pas la meilleure moyenne dans toutes les configurations "
                         "intra- et inter-session.")
    if not 0.4 <= c.kappa("4classes", "inter", "CSP+LDA") <= 0.7:
        problemes.append("Le kappa de CSP+LDA (4 classes, inter-session) sort de la fourchette "
                         "attendue 0,4-0,7 : rechercher une erreur ou une fuite de données.")
    if not c.kappa("binaire", "intra", "EEGNet") < c.kappa("binaire", "intra", "CSP+LDA"):
        problemes.append("EEGNet n'est plus en retrait de CSP+LDA en binaire intra-session.")
    if abs(c.kappa("4classes", "loso", "EEGNet, alignement (réf. session 1)")
           - c.kappa("4classes", "loso", "TS+LR, recentrage (réf. session 1)")) > 0.05:
        problemes.append("EEGNet et TS+LR recentré ne sont plus au même niveau en inter-sujets.")
    if not (c.kappa("4classes", "loso", "TS+LR, recentrage (réf. session 1)")
            > c.kappa("4classes", "loso", "TS+LR, sans recentrage")):
        problemes.append("Le recentrage n'améliore plus TS+LR en inter-sujets.")
    return problemes


def partie_explicative(tableau: pd.DataFrame) -> list:
    """Sections explicatives, dans l'ordre du rapport."""
    c = Chiffres(tableau)
    problemes = verifier_affirmations(c)
    avertissement = []
    if problemes:
        avertissement = ["> **Attention : certaines affirmations du texte ne sont plus vérifiées par les "
                         "données de cette exécution :**"] + [f"> - {p}" for p in problemes] + [""]
        for p in problemes:
            print(f"AVERTISSEMENT : {p}")
    return avertissement + [resume_express(c), "", CONTEXTE, "", METHODE_EVALUATION, "",
            "## 3. Les phases du projet : but, démarche, résultats", "",
            phase0(), "", phase1(), "", phase2(c), "", phase3(c), "", phase4(c), "", phase5(c), "",
            phase6()]
