# Interface cerveau-machine par imagerie motrice : rapport de synthèse

*Projet de Bilal Mokeddem. Document généré automatiquement par `src/synthese.py` à partir des fichiers de `results/` : aucun chiffre n'est saisi à la main. Sauf mention contraire, les valeurs sont des moyennes sur les 9 personnes du jeu BCI Competition IV 2a.*

![Synthèse](../reports/figures/synthese_methodes.png)

## En bref

- **Objectif :** deviner, à partir de l'activité électrique du cerveau (EEG), quel mouvement une personne
  *imagine* (main gauche, main droite, pieds ou langue), sans qu'elle bouge.
- **Le signal existe et il est latéralisé :** imaginer la main droite fait baisser l'activité rythmique au-dessus
  du cortex moteur gauche, et inversement (phase 1).
- **Méthode de référence (CSP + LDA) :** kappa moyen de 0,50 en 4 classes
  (accuracy 62 %, hasard à 25 %), dans les conditions de la compétition.
- **Meilleure méthode : la géométrie riemannienne (TS+LR)**, kappa 0,56
  (accuracy 67 %).
- **Sur une personne jamais vue,** le kappa chute de 0,35 ; un recentrage mathématique sans aucune étiquette
  du nouvel utilisateur en récupère 0,11.
- **Le deep learning (EEGNet)** est en retrait avec peu de données par personne, mais rejoint les meilleures
  méthodes quand on l'entraîne sur plusieurs personnes.
- **La plus grande source de variation, ce sont les personnes elles-mêmes,** pas les algorithmes.

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
une flèche indique le mouvement à imaginer (t = 2 s), la personne l'imagine jusqu'à t = 6 s.

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
  données avant (« p ») et après correction (« p corrigée »).

## 3. Les phases du projet : but, démarche, résultats

### Phase 0 : préparer un environnement reproductible

- **But :** que n'importe qui puisse relancer le projet et obtenir les mêmes chiffres.
- **Démarche :** environnement Python isolé (`.venv`), versions exactes des bibliothèques figées dans
  `requirements.txt`, graine aléatoire fixée à 42, dépôt git, test de chargement d'une première personne.
- **Réflexion :** les bibliothèques d'EEG (MOABB, braindecode) changent souvent d'interface d'une version à l'autre ;
  les noms de fonctions ont donc été vérifiés dans la version installée plutôt que supposés.
- **À retenir :** sans reproductibilité, un résultat n'a pas de valeur scientifique.

### Phase 1 : vérifier que le signal recherché existe

- **But :** avant tout apprentissage automatique, s'assurer que l'imagerie motrice laisse une trace visible et
  conforme à la neurophysiologie (l'ERD latéralisée).
- **Démarche :**
  1. *Vérifier le découpage des essais.* Les marqueurs de MOABB ont été comparés aux fichiers officiels de la
     compétition : décalage constant de 2,000 s (le marqueur est placé sur la
     flèche, pas au début de l'essai), étiquettes identiques pour les 9 personnes, et les essais découpés par MOABB
     coïncident avec le signal brut (corrélation d'au moins 0,97).
     La fenêtre analysée couvre donc bien toute la période d'imagerie. Une erreur de 2 secondes ici aurait faussé tout
     le projet sans aucun message d'erreur.
  2. *Cartes temps-fréquence.* Pour chaque instant et chaque fréquence, on calcule la variation de puissance par
     rapport au repos (avant la flèche) : ERD % = (puissance − repos) / repos × 100. Une valeur négative est une ERD.
  3. *Filtre spatial laplacien.* Le crâne diffuse les signaux : chaque électrode capte un mélange de sources, dont le
     puissant rythme alpha visuel de l'arrière de la tête. Le laplacien soustrait à chaque électrode l'activité de ses
     voisines pour ne garder que la source située juste dessous.
- **Résultats (moyenne des 9 personnes, rythme mu, laplacien) :** en imaginant la **main droite**, la puissance baisse de
  23,2 % sur C3 (hémisphère gauche) contre 10,9 % sur C4 ;
  en imaginant la **main gauche**, de 24,0 % sur C4 contre 19,9 % sur C3.
  La désynchronisation est plus forte du côté opposé à la main, nettement pour la main droite, plus modestement
  pour la main gauche. Ce schéma se retrouve dans
  78 % des cas (personne × bande × main) avec le laplacien, contre 64 % sans.
- **À retenir :** le signal existe et il est latéralisé comme le prévoit la physiologie, mais avec de grandes différences
  entre personnes. Figures : `reports/figures/phase1_*.png`.

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
- **Résultats :** en 4 classes inter-session, kappa 0,50, toutes les personnes
  au-dessus du hasard (9/9).
  En binaire inter-session, kappa 0,57,
  8/9 personnes au-dessus du hasard.
  Ces valeurs sont dans la fourchette publiée pour cette méthode sur ce jeu de données, ce qui écarte une fuite de données.
- **À retenir :** une méthode simple et interprétable fonctionne bien chez certaines personnes et presque pas chez
  d'autres (« illettrisme BCI »). Le passage d'un jour à l'autre dégrade les résultats en 4 classes : c'est la
  **non-stationnarité** du signal (casque replacé, fatigue, impédance des électrodes).

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
- **Résultats (4 classes, inter-session) :** TS+LR obtient un kappa de 0,56, contre
  0,50 pour CSP+LDA et 0,43 pour MDM.
  TS+LR fait mieux que CSP+LDA chez 7 personnes sur 9 (p = 0,039),
  mais après correction pour les 12 tests réalisés, la différence n'est plus significative
  (p corrigée = 0,27).
- **À retenir :** TS+LR a la meilleure moyenne partout, sans aucun réglage. Mais avec 9 personnes, cet avantage reste
  une tendance et non une preuve : il faudrait davantage de personnes pour conclure.

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
- **Résultats (TS+LR, 4 classes) :** modèle propre à la personne 0,56 ; modèle des autres sans recentrage 0,21 ;
  avec recentrage 0,32 (référence session 1) et 0,38 (transductif). Le recentrage récupère
  31 % de la perte en conditions réalistes et
  47 % en transductif.
- **À retenir :** un modèle générique perd beaucoup sur un nouvel utilisateur ; le recentrage en rattrape une part
  importante sans aucune étiquette, mais ne remplace pas une calibration. Il pourrait en revanche la raccourcir.

### Phase 5 : le deep learning (EEGNet)

- **But :** comparer un réseau de neurones conçu pour l'EEG aux méthodes précédentes, dans les mêmes conditions.
- **Démarche :** EEGNet (bibliothèque braindecode) apprend lui-même ses filtres temporels et spatiaux. Pour une
  comparaison équitable, il reçoit exactement les mêmes essais et passe par la même boucle d'évaluation ; ses réglages
  sont ceux de l'article original, fixés à l'avance, sans arrêt précoce (il faudrait des données de validation, et les
  prendre sur le test serait une fuite). Les protocoles les plus lourds ont été calculés sur GPU (Google Colab).
- **Résultats :** avec peu d'essais par personne, EEGNet est en retrait (binaire intra-session : kappa
  0,26 contre 0,56 pour CSP+LDA).
  Le diagnostic l'explique : sur la personne 7, il atteint un kappa de 0,93 à
  0,99 sur les données d'apprentissage mais de 0,01 à
  0,12 en test : il **apprend par cœur** (surapprentissage). Entraîné sur 8 personnes (protocole
  « sujet jamais vu », 4 classes), il obtient 0,33,
  au niveau de TS+LR recentré (0,32).
  L'alignement des signaux l'aide en 4 classes (+0,06 de kappa,
  7/9 personnes), sans que ce soit significatif.
- **À retenir :** le deep learning a besoin de beaucoup de données. Avec quelques centaines d'essais d'une personne,
  les méthodes qui intègrent la connaissance du domaine (la puissance des rythmes mu et beta) sont meilleures et plus
  stables ; avec plusieurs milliers d'essais de plusieurs personnes, EEGNet rejoint les meilleures.

### Phase 6 : synthèse et reproductibilité

- **But :** pouvoir tout relancer d'une seule commande et présenter les résultats sans erreur de recopie.
- **Démarche :** le script `src/tout_relancer.py` enchaîne toutes les étapes ; ce rapport est généré par
  `src/synthese.py` directement à partir des fichiers de résultats.
- **À retenir :** chaque chiffre de ce document provient d'une exécution réelle et sauvegardée dans `results/`.

## 4. Tableaux comparatifs complets

Moyenne ± écart-type sur les 9 personnes. L'écart-type mesure la dispersion entre personnes : un écart-type élevé signifie que la méthode marche très bien chez certaines et mal chez d'autres. La meilleure méthode de chaque tableau est en gras. Pour le protocole « sujet jamais vu », « réf. session 1 » désigne le recentrage réaliste et « transductif » la variante optimiste (voir phase 4).

### 4 classes, intra-session

| Méthode | Kappa (moyenne ± écart-type) | Accuracy (moyenne ± écart-type) |
|---|---|---|
| CSP+LDA | 0,552 ± 0,250 | 0,664 ± 0,188 |
| **TS+LR** | 0,593 ± 0,243 | 0,695 ± 0,182 |
| MDM | 0,468 ± 0,214 | 0,601 ± 0,160 |
| EEGNet | 0,457 ± 0,249 | 0,593 ± 0,187 |

### 4 classes, inter-session

| Méthode | Kappa (moyenne ± écart-type) | Accuracy (moyenne ± écart-type) |
|---|---|---|
| CSP+LDA | 0,499 ± 0,229 | 0,624 ± 0,172 |
| **TS+LR** | 0,562 ± 0,225 | 0,671 ± 0,169 |
| MDM | 0,432 ± 0,188 | 0,574 ± 0,141 |
| EEGNet | 0,477 ± 0,226 | 0,608 ± 0,170 |

### 4 classes, sujet jamais vu

| Méthode | Kappa (moyenne ± écart-type) | Accuracy (moyenne ± écart-type) |
|---|---|---|
| CSP+LDA, sans recentrage | 0,251 ± 0,211 | 0,438 ± 0,158 |
| TS+LR, sans recentrage | 0,211 ± 0,164 | 0,408 ± 0,123 |
| MDM, sans recentrage | 0,127 ± 0,155 | 0,345 ± 0,116 |
| EEGNet, sans alignement | 0,267 ± 0,228 | 0,451 ± 0,171 |
| TS+LR, recentrage (réf. session 1) | 0,319 ± 0,211 | 0,490 ± 0,158 |
| MDM, recentrage (réf. session 1) | 0,276 ± 0,181 | 0,457 ± 0,136 |
| EEGNet, alignement (réf. session 1) | 0,327 ± 0,212 | 0,495 ± 0,159 |
| **TS+LR, recentrage transductif** | 0,377 ± 0,224 | 0,532 ± 0,168 |
| MDM, recentrage transductif | 0,317 ± 0,175 | 0,488 ± 0,131 |

### Main gauche contre main droite, intra-session

| Méthode | Kappa (moyenne ± écart-type) | Accuracy (moyenne ± écart-type) |
|---|---|---|
| CSP+LDA | 0,561 ± 0,336 | 0,780 ± 0,168 |
| **TS+LR** | 0,611 ± 0,303 | 0,806 ± 0,151 |
| MDM | 0,515 ± 0,323 | 0,757 ± 0,161 |
| EEGNet | 0,259 ± 0,303 | 0,630 ± 0,151 |

### Main gauche contre main droite, inter-session

| Méthode | Kappa (moyenne ± écart-type) | Accuracy (moyenne ± écart-type) |
|---|---|---|
| CSP+LDA | 0,571 ± 0,331 | 0,785 ± 0,166 |
| **TS+LR** | 0,591 ± 0,294 | 0,796 ± 0,147 |
| MDM | 0,528 ± 0,339 | 0,764 ± 0,169 |
| EEGNet | 0,426 ± 0,363 | 0,713 ± 0,182 |

### Main gauche contre main droite, sujet jamais vu

| Méthode | Kappa (moyenne ± écart-type) | Accuracy (moyenne ± écart-type) |
|---|---|---|
| CSP+LDA, sans recentrage | 0,327 ± 0,293 | 0,664 ± 0,147 |
| TS+LR, sans recentrage | 0,378 ± 0,284 | 0,689 ± 0,142 |
| MDM, sans recentrage | 0,170 ± 0,174 | 0,585 ± 0,087 |
| EEGNet, sans alignement | 0,370 ± 0,285 | 0,685 ± 0,142 |
| TS+LR, recentrage (réf. session 1) | 0,392 ± 0,256 | 0,696 ± 0,128 |
| MDM, recentrage (réf. session 1) | 0,369 ± 0,202 | 0,684 ± 0,101 |
| EEGNet, alignement (réf. session 1) | 0,386 ± 0,245 | 0,693 ± 0,122 |
| **TS+LR, recentrage transductif** | 0,448 ± 0,307 | 0,724 ± 0,154 |
| MDM, recentrage transductif | 0,395 ± 0,250 | 0,698 ± 0,125 |

## 5. Niveau du hasard (CSP+LDA, 100 permutations des étiquettes)

Le « 95e centile » est le score que le hasard ne dépasse que 5 fois sur 100 : un vrai score au-dessus de cette valeur est significatif.

| Tâche | Évaluation | Hasard théorique | 95e centile des permutations (moyenne des sujets) | Sujets au-dessus du hasard (p < 0,05) |
|---|---|---|---|---|
| 4 classes | Intra-session | 25 % | 0,297 | 9/9 |
| 4 classes | Inter-session | 25 % | 0,340 | 9/9 |
| Main gauche contre main droite | Intra-session | 50 % | 0,588 | 6/9 |
| Main gauche contre main droite | Inter-session | 50 % | 0,626 | 8/9 |

## 6. Meilleures et moins bonnes personnes (4 classes, inter-session, kappa)

| Sujet | CSP+LDA | TS+LR | MDM | EEGNet | moyenne des méthodes |
|---|---|---|---|---|---|
| 3 | 0,690 | 0,819 | 0,602 | 0,759 | 0,718 |
| 1 | 0,708 | 0,778 | 0,657 | 0,643 | 0,697 |
| 9 | 0,745 | 0,736 | 0,556 | 0,708 | 0,686 |
| 8 | 0,713 | 0,690 | 0,648 | 0,662 | 0,678 |
| 7 | 0,500 | 0,694 | 0,333 | 0,403 | 0,483 |
| 4 | 0,449 | 0,454 | 0,412 | 0,389 | 0,426 |
| 6 | 0,292 | 0,319 | 0,255 | 0,398 | 0,316 |
| 2 | 0,250 | 0,306 | 0,171 | 0,194 | 0,230 |
| 5 | 0,143 | 0,259 | 0,255 | 0,139 | 0,199 |

Meilleurs sujets : **3, 1, 9** ; sujets les plus difficiles : **6, 2, 5**. L'écart de kappa moyen entre le meilleur et le moins bon sujet est de 0,519, bien plus grand que l'écart entre la meilleure et la moins bonne méthode (0,130). Le classement des sujets est très stable d'une méthode à l'autre (corrélation de Spearman moyenne entre méthodes : 0,83). La difficulté tient donc au sujet plus qu'à l'algorithme : c'est la signature de l'« illettrisme BCI », une partie des utilisateurs ne produisant pas de désynchronisation assez nette pour être décodée.

## 7. Tests statistiques principaux (Wilcoxon apparié, 9 personnes)

Lecture : « 7/9 sujets améliorés » signifie que la première méthode fait mieux chez 7 personnes sur 9. Une p corrigée inférieure à 0,05 indique une différence significative malgré le nombre de tests réalisés.

- **TS+LR contre CSP+LDA**, 4 classes, inter-session : différence de kappa 0,063, 7/9 sujets améliorés, p = 0,0391, p corrigée (Holm, 12 tests) = 0,2734.
- **TS+LR contre CSP+LDA**, main gauche contre main droite, inter-session : différence de kappa 0,020, 6/9 sujets améliorés, p = 0,5703, p corrigée (Holm, 12 tests) = 1,0000.
- **TS+LR contre MDM**, 4 classes, inter-session : 9/9 sujets améliorés, p = 0,0039, p corrigée = 0,0469.
- **TS+LR, 4 classes, perte quand le sujet n'a jamais été vu** : différence de kappa −0,351, 0/9 sujets en faveur de « aucun », p = 0,0039, p corrigée (Holm, 12 tests) = 0,0469.
- **TS+LR, 4 classes, gain du recentrage (référence : session 1)** : différence de kappa 0,108, 8/9 sujets en faveur de « calibration », p = 0,0078, p corrigée (Holm, 12 tests) = 0,0625.
- **TS+LR, 4 classes, gain du recentrage transductif** : différence de kappa 0,166, 9/9 sujets en faveur de « transductif », p = 0,0039, p corrigée (Holm, 12 tests) = 0,0469.
- Part de la perte inter-sujets récupérée par le recentrage (calibration, 4 classes, TS+LR) : 31 %.
- Part de la perte inter-sujets récupérée par le recentrage (transductif, 4 classes, TS+LR) : 47 %.
- **EEGNet contre CSP+LDA**, 4 classes, intra-session : différence de kappa −0,095, 0/9 sujets en faveur d'EEGNet, p = 0,0039, p corrigée (Holm, 18 tests) = 0,0703.
- **EEGNet contre CSP+LDA**, 4 classes, inter-session : différence de kappa −0,022, 2/9 sujets en faveur d'EEGNet, p = 0,4961, p corrigée (Holm, 18 tests) = 1,0000.
- **EEGNet contre CSP+LDA**, 4 classes, sujet jamais vu : différence de kappa 0,076, 7/9 sujets en faveur d'EEGNet, p = 0,0547, p corrigée (Holm, 18 tests) = 0,4922.

## 8. Limites

- **9 sujets seulement.** Avec 9 sujets, la plus petite p-valeur possible au test de Wilcoxon est
  2/512, soit environ 0,004 : après correction pour comparaisons multiples, seules les différences
  observées chez les 9 sujets peuvent être déclarées significatives. Plusieurs tendances (TS+LR
  meilleur que CSP+LDA, EEGNet meilleur que CSP+LDA en inter-sujets) restent donc non démontrées.
- **Un seul jeu de données** (BCI Competition IV 2a) : les conclusions demandent à être confirmées
  sur d'autres jeux de données d'imagerie motrice disponibles dans MOABB.
- **Analyse hors ligne.** Le filtrage passe-bande est appliqué sur l'enregistrement continu, avec un
  filtre non causal (qui utilise des échantillons futurs) : c'est sans fuite d'étiquettes, mais un
  système en temps réel utiliserait un filtre causal.
- **Essais marqués comme artefacts conservés.** MOABB renvoie les 288 essais de chaque session, y compris
  ceux que les organisateurs ont signalés comme contaminés ; les scores incluent donc ces essais.
- **Réglages fixés a priori, sans optimisation.** Fenêtre (0,5 à 2,5 s), bande (8-30 Hz), nombre de
  composantes CSP, hyperparamètres d'EEGNet : choisis avant toute évaluation. C'est un choix de rigueur
  (aucune sélection sur le test), mais toutes les méthodes pourraient gagner à un réglage par validation
  croisée imbriquée sur les données d'apprentissage.
- **Niveau du hasard par permutation calculé en phase 2 seulement** (100 permutations, coût de calcul) ;
  les phases suivantes comparent les méthodes par tests appariés.
- **EEGNet : une seule graine par entraînement, matériel mixte.** Le diagnostic montre jusqu'à 0,25 de
  kappa d'écart entre graines sur un même sujet ; l'inter-session a été calculé sur CPU (PyTorch 2.14),
  l'intra-session et le protocole inter-sujets sur GPU Colab (PyTorch 2.11). Deux exécutions sur CPU
  peuvent différer d'un essai (calculs parallèles non strictement déterministes).
- **Recentrage transductif optimiste.** Il calcule la référence du nouveau sujet sur ses essais de test
  (sans étiquettes) ; la variante réaliste est celle dont la référence est la session 1.

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
| **Wilcoxon (test de)** | Test statistique non paramétrique pour comparer deux méthodes sur les mêmes personnes. |
