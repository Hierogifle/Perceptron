# Building Perceptron

> "The perceptron is capable of generalization and abstraction; it may recognize similarities between patterns which are not identical." - Frank Rosenblatt

# 📖 Contexte du Projet
Ce projet s'inscrit dans le cadre d'une initiation au Deep Learning à La Plateforme_. Il vise à comprendre et implémenter le Perceptron, le premier neurone artificiel inventé par Frank Rosenblatt, considéré comme la pierre angulaire de l'apprentissage automatique moderne.

L'objectif principal est de développer une compréhension approfondie des concepts fondamentaux du Machine Learning à travers l'implémentation d'un Perceptron en Python, puis de tester ce modèle sur un cas d'usage réel avec le dataset Breast Cancer Wisconsin.

# 🎯 Objectifs du Projet
## Phase 1 : Fondements Théoriques

- Définir et comparer les notions de Machine Learning et Deep Learning

- Identifier 3 applications concrètes du Deep Learning

- Analyser le lien entre neurone biologique et Perceptron

- Comprendre la fonction mathématique et les règles d'apprentissage du Perceptron

## Phase 2 : Implémentation

- Développer une classe Perceptron en Python (programmation orientée objet)

- Tester le modèle sur des données factices générées aléatoirement

- Valider le fonctionnement de l'algorithme d'apprentissage

## Phase 3 : Application Pratique

- Analyser le dataset Breast Cancer Wisconsin

- Réaliser une analyse exploratoire complète des données

- Appliquer des techniques de réduction de dimensionnalité

- Évaluer les performances du Perceptron avec des métriques appropriées

- Proposer des améliorations pour optimiser les résultats

# 📊 Données Utilisées

**Dataset** : Breast Cancer Wisconsin
**Source** : UCI Machine Learning Repository

- Problématique : Classification binaire pour le diagnostic du cancer du sein

- Caractéristiques : 30 features numériques décrivant les caractéristiques des noyaux cellulaires

- Objectif : Prédire si une tumeur est bénigne ou maligne

## Approche d'Analyse

- Nettoyage des données : Gestion des valeurs manquantes et des outliers

- Analyse exploratoire : Visualisations et statistiques descriptives

- Réduction de dimensionnalité : Application de techniques appropriées

- Modélisation : Entraînement du Perceptron développé

- Évaluation : Métriques de performance adaptées au contexte médical

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

text
building-perceptron/
├── script.py                 # Classe Perceptron implémentée
├── notebook.ipynb           # Analyse complète et modélisation
├── README.md               # Documentation du projet
├── data/                   # Données du projet
├── visualizations/         # Graphiques et visualisations
└── requirements.txt        # Dépendances Python

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