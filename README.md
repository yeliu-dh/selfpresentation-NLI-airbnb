> 🇫🇷 [Français](#version-française) · 🇬🇧 [English](#english-version) · 🇨🇳 [中文](#中文)


# Version française:
#  Présentation de soi et performance des hôtes sur Airbnb : une approche psychométrique par NLI zero-shot
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


# English version
# Self-Presentation and Host Performance on Airbnb: A Psychometric Approach Using Zero-Shot NLI

## 🔍 Abstract

This research examines the impact of Airbnb hosts’ self-presentation tactics on their monthly booking rate and the moderating role of Superhost status, using a 22-item psychometric scale applied at scale through zero-shot NLI.

## 🏷️ 5 Tactics Adapted to the Airbnb Context

* **Relational tactics**: openness · sociability · authenticity
* **Promotional tactics**: self-promotion · exemplarity

## 📊 Dataset

* **Source**: [Inside Airbnb](https://insideairbnb.com/get-the-data/)
* **Sample**: Airbnb listings in Paris — December 2023
* **Size**: ~75,000 listings
* **Key variable**: host profile description (`host_about`)

## 🔬 Methodology

### Scale Construction and Psychometric Validation

* **EFA** (*Exploratory Factor Analysis*)
* **Psychometric validation**: Cronbach’s α · communalities · complexity
* **Validated scale**: 22 descriptive items → 5 dimensions

### Natural Language Inference (NLI)

* **Multilingual zero-shot NLI**: item-level evaluation of the semantic presence of each item in the host descriptions
* **Model**: [BGE-M3-ZeroShot-v2.0](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0)
* **Weighted scores**: 22 item scores → 5 dimension scores

### Econometric Analysis

* **Tactics & monthly booking rate**: OLS regression
* **Superhosts vs. other hosts**: OLS with interaction term · t-test · Cohen’s *d*

## 💡 Main Results

* **Prevalence**: Self-promotion and exemplarity are the most common tactics.

* **Impact**: Sociability and authenticity increase the booking rate, while exemplarity has a negative effect. The other tactics show no significant effects.

* **Moderation**: Superhost status reveals the negative effect of self-promotion but may attenuate the negative effect of exemplarity. The effects of the other tactics remain unchanged.

## 🗂️ Project Structure

* **`scripts/`**: `.ipynb` notebooks for preprocessing, factor analysis, and modeling
* **`utils/`**: utility functions
* **`*_results/`**: results from the different analyses and processing steps

## ⚙️ Reproduction

* **Python**: `3.10`
* **Dependencies**: `pip install -r requirements.txt`
* **Data**: not included in the repository; reproduction scripts are provided. Data available upon request.
