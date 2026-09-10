#  Présentation de soi et performance des hôtes sur Airbnb : une approche psychométrique par NLI zero-shot

> **Note** — Première partie de mon mémoire de master.  
> 🇫🇷 [Français](#français) · 🇬🇧 [English](#english) · 🇨🇳 [中文](#中文)


## Français:
## 🔍 Résumé
Cette recherche étudie l’impact des tactiques de présentation de soi des hôtes Airbnb sur leur taux de réservation et le rôle modérateur du statut de Superhôte, à partir d’une échelle psychométrique de 22 items appliquée à grande échelle par NLI zero-shot.

## 🏷️ 5 tactiques adaptées au contexte d’Airbnb:
- **tactiques relationnelles** : Ouveture, sociabilité, authenticité
- **tactiques promotionnelles** : Auto-promotion, exemplarité

## 📊 Jeu de données 
- Source : [Inside Airbnb](https://insideairbnb.com/get-the-data/)  
- Échantillon : logements Airbnb à Paris en décembre 2023
- Taille : ~ 75 000 annonces
- Variable clé : description personnelle de l'hôte (*host_about*)

## 🔬 Méthodologie 
### Construction de l'échelle
- EFA (Exploratory Factor Analysis)
- Validation psychométrique: Cronbash's α, communalités, complexité
- Échelle validée: 5 tactiques → 22 items descriptifs

### Inférence en langage naturel (NLI)
- Classification zéro-shot mutilingue: évaluer dans quelle mesure chaque item est sémantiquement présent dans un texte donnée 
- Modèle : [bge-m3-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0)
- Scores pondérés : 22 scores d’items → 5 scores dimensionnels

### Analyse économétrique 
- Tactiques & taux de réservation mensuel : régression OLS  
- Superhôte vs. Autres : OLS avec terme d'intéraction, t-test, Cohen's *d*

## 💡 Résultats principaux
- **Prévalence** : L’auto-promotion et l’exemplarité sont les plus courantes;
- **Impact** :La sociabilité et l’authenticité augmentent le taux de réservation, alors que l’exemplarité a un effet négatif, d'autres ne présentent pas d'effet significatif; 
- **Modération** : Le statut de Superhôte joue un rôle modérateur : il révèle l'effet négatif de l'auto-promotion, mais peut atténuer l'effet négatif de l'exemplarité. Les effets des autres tactiques restent inchangés.

## 🗂️ Structure du Projet
- scripts : fichier `.ipynb` pour le prétraitement, l'analyse factorielle et la modélisation
- utils : fonctions utilitaires
- *_results : résultats des traitements et des analyses

## ⚙️ Reproduction

- **Python** : `3.10`
- **Dépendances** : `pip install -r requirements.txt`
- **Données** : non incluses ; scripts de reproduction fournis. Disponibles sur demande.
