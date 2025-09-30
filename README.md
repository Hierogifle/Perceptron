# Building Perceptron

> "The perceptron is capable of generalization and abstraction; it may recognize similarities between patterns which are not identical." - Frank Rosenblatt

# 📖 Contexte du Projet
Ce projet s'inscrit dans le cadre d'une initiation au Deep Learning à La Plateforme_. Il vise à comprendre et implémenter le Perceptron, le premier neurone artificiel inventé par Frank Rosenblatt, considéré comme la pierre angulaire de l'apprentissage automatique moderne.

L'objectif principal est de développer une compréhension approfondie des concepts fondamentaux du Machine Learning à travers l'implémentation d'un Perceptron en Python, puis de tester ce modèle sur un cas d'usage réel avec le dataset Breast Cancer Wisconsin.

# 🎯 Objectifs du Projet

## Phase 1 : Fondements Théoriques

### Machine Learning vs. Deep Learning : définitions et comparaison

#### Machine Learning (Apprentissage Automatique)

Le Machine Learning regroupe des méthodes permettant à une machine d’apprendre à partir de données sans être explicitement programmée pour chaque tâche. Les algorithmes classiques incluent :

- Régressions linéaires et logistiques
- Arbres de décision et forêts aléatoires
- SVM (Support Vector Machines)
- k-nearest neighbors

Ils reposent souvent sur des caractéristiques (features) ingénieusement extraites et nécessitent un prétraitement humain des données (sélection, transformation, normalisation).

#### Deep Learning (Apprentissage Profond)

Le Deep Learning est un sous-ensemble du Machine Learning qui utilise des réseaux de neurones profonds (plusieurs couches de neurones artificiels) capables d’apprendre automatiquement les représentations des données à différents niveaux d’abstraction (features hiérarchiques).

Aspect                            | Machine              | Learning Deep Learning              |
----------------------------------|----------------------|-------------------------------------|
Caractéristiques                  | Conception manuelle  | Apprentissage automatique           |
Complexité des données            | Modérée              | Très élevée (images, sons, textes…) |
Quantité de données               | Moyenne              | Grandes quantités requises          |
Puissance de calcul               | Ordinateur classique | GPU/TPU souvent nécessaires         |
Performances sur tâches complexes | Limitées             | Excellentes                         |

### Quand utiliser l’un plutôt que l’autre ?

#### Machine Learning

- Jeux de données petits à moyens
- Problèmes bien compris où les features manuelles sont efficaces
- Contraintes de puissance de calcul

#### Deep Learning

- Données volumineuses et non structurées (images, audio, texte)
- Besoin d’extraire des représentations complexes sans expertise métier poussée
- Disponibilité de ressources GPU et tolérance à des temps d’entraînement plus longs

### Applications du Deep Learning

Voici trois exemples emblématiques, inspirés d’AI Experiments et d’OpenAI :

#### 1. Diagnostic médical assisté par imagerie

Des réseaux de neurones convolutionnels (CNN) analysent des radiographies ou IRM pour détecter automatiquement des tumeurs ou anomalies. Cette approche accélère le diagnostic du cancer et améliore la précision des dépistages.

#### 2. Véhicules autonomes

Les systèmes embarqués s’appuient sur des CNN et des vision transformers pour reconnaître en temps réel les piétons, panneaux de signalisation et autres véhicules, permettant à la voiture de naviguer en toute sécurité sans intervention humaine.

#### 3. Traitement automatique du langage naturel (NLP)

Les modèles de type Transformers (GPT, BERT) gèrent la traduction, la génération de texte et les chatbots. Par exemple, ChatGPT comprend et produit un langage proche du dialogue humain pour l’assistance en ligne ou la création de contenu.

### Le Perceptron

##### Introduction

Le Perceptron est le premier modèle de neurone artificiel, mis au point par Frank Rosenblatt en 1957. Il sert de brique de base à l’apprentissage profond. Ce dossier explique pas à pas le fonctionnement du perceptron, ses limites, puis montre comment l’étendre et l’utiliser dans des contextes plus complexes.

#### 1. **Qu’est-ce qu’un Perceptron ?**

Un perceptron est un programme informatique qui imite grossièrement le comportement d’un neurone du cerveau :

- **Neurone biologique** : capte des signaux via des extensions appelées dendrites, les additionne dans le corps cellulaire, et, si le total dépasse un seuil, génère un signal électrique (potentiel d’action) transmis à d’autres neurones.

- **Perceptron** : reçoit des entrées numériques x1,x2,…,xn, multiplie chacune par un poids w1,w2,…,wn, ajoute un biais b, puis décide d’envoyer un “oui” (1) ou un “non” (0) selon le résultat.

Cette structure lui permet de distinguer deux catégories d’objets ou de données quand elles sont séparables par une ligne droite (en deux dimensions) ou un plan (en plusieurs dimensions).

#### 2. **Formule mathématique et signification des termes**

La décision du perceptron se calcule ainsi :

<div align="center">
   <img src="images/formule_perceptron.png" alt="y_hat = activation(w1 * x1 + w2 * x2 + ... + wn * xn + b)" style="width: 40%; max-width: 900px;">
</div>

- ŷ  : prédiction du perceptron
- z : somme pondérée des entrées plus le biais (équivalent du potentiel d’action).
- f : fonction d’activation qui convertit z en 0 ou 1.
- wi : poids attribué à cette entrée (force du lien synaptique).
- xi : valeur de la i-ème entrée (ex. intensité d’un pixel).
- b : biais, permet de décaler la frontière de décision.

La fonction d’activation la plus simple est le seuil de Heaviside :

<div align="center">
   <img src="images/regle_activation.png" alt="Fonction d'activation" style="width: 30%; max-width: 900px;">
</div>

#### 3. Comment apprend-on ?

L’apprentissage consiste à ajuster les poids wi et le biais b pour que le perceptron donne la bonne réponse sur des exemples connus. on utlise la règle d'apprentissage du perceptron (mise à jour des poids) :

<div align="center">
   <img src="images/regle_apprentissage.png" alt="Fonction d'apprentissage" style="width: 30%; max-width: 900px;">
</div>

- η : taux d'apprentissage
- y : label attendu

Ce formalisme est la base du perceptron simple, utilisé pour la classification binaire linéaire.

1. Calculer la sortie : ŷ = f(∑wi × xi + b)
2. Mesurer l'erreur : δ = y - ŷ  
3. Mettre à jour les poids : wi = wi + η × δ × xi
4. Mettre à jour le biais : b = b + η × δ
    où η est le taux d’apprentissage, un petit nombre (ex. 0,01) qui détermine la vitesse d’ajustement.

On répète ces étapes sur tous les exemples, plusieurs fois (plusieurs époques), jusqu’à obtenir un taux de bonnes réponses satisfaisant.

#### 4. Processus d’entraînement complet

1. Initialisation aléatoire : attribuer de petits poids et biais proches de zéro.
2. Boucle d’époques :
      - Pour chaque exemple :
            - Prédire y^
            - Calculer l’erreur δ
            - Ajuster wi et b

3. Vérification : arrêter si l’erreur moyenne tombe en-dessous d’un seuil ou après un nombre fixe d’époques.

#### 5. Limites du Perceptron

- Données linéairement séparables : il ne résout que les problèmes où une ligne ou un plan peut séparer les deux classes (ex. XOR n’est pas séparable).
- Convergence non garantie si les données ne satisfont pas cette condition.
- Frontière plane : incapable de modéliser des formes de séparation complexes.
- Sensibilité aux choix du taux d’apprentissage et à l’initialisation des poids.

#### 6. Aller plus loin : perceptrons multicouches et backpropagation

Pour traiter des données non linéaires, on assemble plusieurs perceptrons en couches :
- Chaque couche applique la même opération (somme pondérée + fonction d’activation) sur la sortie de la couche précédente.
- On appelle cela un réseau de neurones multicouches (MLP).
- L’algorithme de rétropropagation (backpropagation) calcule les dérivées de l’erreur par rapport à chaque poids en propageant l’erreur à rebours, puis ajuste tous les poids simultanément.

#### 7. Versions améliorées et bonnes pratiques

- Fonctions d’activation modernes : ReLU, tanh, sigmoïde, qui gèrent mieux le phénomène de gradient.
- Optimiseurs avancés : Adam, RMSProp, qui adaptent automatiquement le taux d’apprentissage.
- Régularisation : Dropout, weight decay, pour éviter le surapprentissage.
- Normalisation : Batch Normalization, pour stabiliser et accélérer l’entraînement.

#### 8. Exemples de code Python (orienté objet)

*python*

```
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, lr=0.01):
        self.w = np.random.randn(n_inputs) * 0.01
        self.b = 0.0
        self.lr = lr

    def activation(self, z):
        return 1 if z >= 0 else 0

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return self.activation(z)

    def train(self, X_train, y_train, epochs=100):
        for _ in range(epochs):
            for x, y in zip(X_train, y_train):
                y_pred = self.predict(x)
                delta = y - y_pred
                self.w += self.lr * delta * x
                self.b += self.lr * delta
```

#### Test rapide :

**Données factices**

*python*

```
X = np.random.randn(200, 2)
y = np.where(X[:,0] + X[:,1] > 0, 1, 0)

model = Perceptron(n_inputs=2, lr=0.01)
model.train(X, y, epochs=50)

preds = np.array([model.predict(x) for x in X])
print("Précision :", np.mean(preds == y))

<div style="page-break-after: always;"></div>
```

## Phase 2 : Compréhension et Analyse des Données

Statistiques descriptives, visualisations, corrélations

Détection des outliers et valeurs manquantes

### 📊 Données Utilisées

**Dataset** : Breast Cancer Wisconsin
**Source** : UCI Machine Learning Repository (https://archive.ics.uci.edu/datasets)

- Problématique : Classification binaire pour le diagnostic du cancer du sein

- Caractéristiques : 30 features numériques décrivant les caractéristiques des noyaux cellulaires

Structure du Dataset Breast Cancer Wisconsin

### 🔍 Vue d'ensemble

- 569 échantillons (212 malins (37%), 357 bénins(63%))
- 32 colonnes : 1 ID + 1 diagnostic + 30 caractéristiques numériques

Source : Images FNA (biopsie à l'aiguille fine) de masses mammaires

Les caractéristiques sont calculées à partir d'une image numérisée d'une ponction à l'aiguille fine (PAF) d'une masse mammaire. Elles décrivent les caractéristiques des noyaux cellulaires présents sur l'image.

Objectif : Classification binaire pour diagnostic du cancer du sein

### 📊 Variables du dataset

Variable  |	Description                        | Utilité        |
----------|------------------------------------|----------------|
id	      |Identifiant unique de l'échantillon | Identification |
diagnosis |	Diagnostic (M = malin, B = bénin)  | Variable cible |

### 🧬 Les 30 caractéristiques (features)

Chaque caractéristique de base est calculée sous 3 formes différentes :

Suffixes des colonnes :
- _mean (colonnes 3-12) : Valeurs moyennes
- _se (colonnes 13-22) : Erreurs standard (variabilité)
- _worst (colonnes 23-32) : Pires valeurs (moyenne des 3 plus extrêmes)

10 caractéristiques de base mesurées sur le noyau cellulaire :

*radius : Distance moyenne centre-périmètre (taille)*
Définition : Distance moyenne entre le centre du noyau cellulaire et tous les points de son périmètre.
- Calcul concret : On identifie le centre géométrique du noyau, puis on mesure la distance à chaque point du contour et on fait la moyenne.
- Valeurs typiques : 6-28 unités (pixels)
- Signification clinique : Un rayon élevé indique des cellules plus grosses, souvent associées à la malignité.
- Pourquoi c'est important : Les cellules cancéreuses ont tendance à être plus volumineuses que les cellules normales.

*texture : Écart-type des niveaux de gris (rugosité)*
Définition : Écart-type des intensités de niveaux de gris dans une région du noyau.
- Calcul concret : On analyse la variation de luminosité pixel par pixel dans le noyau (homogène vs hétérogène).
- Valeurs typiques : 9-40 unités
- Signification clinique : Texture élevée = noyau "granuleux" ou irrégulier, caractéristique des cellules malignes.
- Analogie : Comme la différence entre une surface lisse (faible texture) et rugueuse (forte texture).

*perimeter : Longueur du contour (circonférence)*
Définition : Longueur totale du contour du noyau cellulaire.
- Calcul concret : Somme des distances entre tous les pixels adjacents formant le contour.
- Valeurs typiques : 40-190 unités
- Signification clinique : Périmètre élevé peut indiquer une forme irrégulière ou une taille importante.
- Relation : Fortement corrélé au rayon et à l'aire.

*area : Surface du noyau (taille 2D)*
Définition : Surface totale occupée par le noyau cellulaire.
- Calcul concret : Nombre de pixels à l'intérieur du contour du noyau.
- Valeurs typiques : 140-2500 unités²
- Signification clinique : Aire importante = cellule volumineuse, souvent maligne.
- Note : Aire = π × rayon², d'où la forte corrélation entre ces variables.

*smoothness : Variation locale du rayon (régularité)*
Définition : Variation locale des longueurs de rayon (écart-type des rayons).
- Calcul concret : On mesure plusieurs rayons depuis le centre et on calcule leur variabilité.
- Valeurs typiques : 0.05-0.16 unités
- Signification clinique : Faible lissage = contour irrégulier, "bosselé", typique des cellules malignes.
- Interprétation : 0 = cercle parfait, valeurs élevées = forme très irrégulière.

*compactness : Périmètre²/aire - 1.0 (forme)*
Définition : Formule (périmètre² / aire) - 1.0
- Calcul concret : Mesure de l'efficacité géométrique de la forme.
- Valeurs typiques : 0.02-0.35 unités
- Signification clinique : Compacité élevée = forme étalée ou irrégulière.
- Référence : Un cercle parfait a une compacité de 0, les formes irrégulières ont des valeurs plus élevées.

*concavity : Sévérité des portions concaves (creux)*
Définition : Sévérité des portions concaves du contour (zones qui "rentrent vers l'intérieur").
- Calcul concret : Somme des profondeurs des indentations divisée par la surface.
- Valeurs typiques : 0-0.43 unités
- Signification clinique : Concavité élevée = nombreux "creux" dans le contour, signe de malignité.
- Visualisation : Comme mesurer la profondeur des "entailles" dans le contour.

*concave_points : Nombre de portions concaves (irrégularité)*
Définition : Nombre de portions concaves sur le contour du noyau.
- Calcul concret : Comptage des zones où le contour "rentre vers l'intérieur".
- Valeurs typiques : 0-0.2 unités (normalisé par la taille)
- Signification clinique : Plus il y a de points concaves, plus la forme est irrégulière et potentiellement maligne.
- Différence avec concavity : Ici on compte, là-bas on mesure la profondeur.

*symmetry : Mesure de symétrie du noyau*
Définition : Mesure de la symétrie du noyau par rapport à son centre.
- Calcul concret : Comparaison entre les moitiés droite/gauche et haut/bas du noyau.
- Valeurs typiques : 0.1-0.3 unités
- Signification clinique : Asymétrie élevée = forme déséquilibrée, souvent associée à la malignité.
- Interprétation : 0 = parfaitement symétrique, valeurs élevées = très asymétrique.

*fractal_dimension : Complexité du contour (dimension fractale)*
Définition : Mesure mathématique de la complexité du contour ("approximation de ligne de côte" - 1).
- Calcul concret : Utilise l'algorithme de "coastline approximation" pour mesurer la rugosité du contour.
- Valeurs typiques : 1.0-2.1 unités
- Signification clinique : Dimension fractale élevée = contour très complexe et irrégulier, caractéristique des cellules malignes.
- Analogie : Comme mesurer la complexité d'une côte : plus elle est découpée, plus sa dimension fractale est élevée.

### 🎯 Variables les plus discriminantes

- *Taille* : radius_mean, area_mean, perimeter_mean
- *Forme* : concavity_mean, concave_points_mean, compactness_mean
- Variables *_worst* : souvent très informatives pour détecter les cas extrêmes

Cette structure permet d'analyser finement les caractéristiques morphologiques des cellules pour distinguer les tumeurs bénignes des malignes.



## Phase 3 : Préparation des Données

Nettoyage des données

Normalisation/standardisation

Réduction de dimensionnalité (PCA)

Division train/test

## Phase 4 : Implémentation du Modèle

Développer la classe Perceptron

Tests sur données factices pour validation

## Phase 5 : Application et Évaluation

Entraînement sur les vraies données préparées

Évaluation des performances

Analyse des résultats et améliorations



# 🛠️ Outils et Technologies

## Langages et Frameworks

- Python : Langage principal du projet

- NumPy : Calculs numériques et algèbre linéaire

- Pandas : Manipulation et analyse de données

- Matplotlib/Seaborn : Visualisation de données

- Scikit-learn : Preprocessing et métriques d'évaluation

# Techniques Appliquées

## Programmation Orientée Objet : Architecture modulaire du Perceptron

- Analyse Exploratoire : Compréhension approfondie des données

- Réduction de Dimensionnalité : Optimisation des features

- Validation Croisée : Évaluation robuste du modèle

# 📁 Structure du Repository

```
text
building-perceptron/
├── .env                    # environnement python
├── script.py                 # Classe Perceptron implémentée
├── notebook.ipynb           # Analyse complète et modélisation
├── README.md               # Documentation du projet
├── Data/                   # Données du projet
├── visualizations/         # Graphiques et visualisations
└── requirements.txt        # Dépendances Python
```

# 📈 Résultats et Conclusions

## Performance du Perceptron

[Insérer ici les métriques obtenues lors de l'évaluation]

## Limites Identifiées

Le Perceptron présente certaines limitations inhérentes :

- Capacité limitée aux problèmes linéairement séparables

- Sensibilité à l'initialisation des poids

- Convergence non garantie pour certains datasets

## Améliorations Proposées

- Implémentation du Perceptron multi-couches

- Utilisation de fonctions d'activation non-linéaires

- Application de techniques de régularisation

- Optimisation des hyperparamètres

# 📚 Bibliographie
## Ressources Académiques

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"

- Minsky, M., & Papert, S. (1969). "Perceptrons: An Introduction to Computational Geometry"

## Ressources Techniques

- Perceptron Algorithm with Code Example ML for beginners!

- Gradient Descent Simply Explained! ML for beginners with Code Example!

- Boruta-py Documentation

## Datasets
- Breast Cancer Wisconsin (Diagnostic) Dataset - UCI Machine Learning Repository

# 🔄 Développement et Maintenance

Ce projet a été développé dans le cadre de la formation en Intelligence Artificielle et Data Science à La Plateforme. Il constitue une base solide pour l'apprentissage des concepts fondamentaux du Machine Learning et peut être étendu avec des algorithmes plus complexes.