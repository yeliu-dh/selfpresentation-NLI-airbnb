#  L'impact des tactiques de présentation de soi sur la performance des transactions au sein des platformes C2C

## Résumé
Fondée sur la théorie de la présentation de soi et de la gestion des impressions de Goffman, cette recherche étudie les stratégies de présentation de soi des hôtes Airbnb à travers leurs descriptions personnelles. Elle développe une échelle de mesure psychométrique pour mesurer les cinq tactiques de présentation de soi des hôtes, puis l’applique à grande échelle à l’aide d’un modèle de classification zero-shot. Les dimensions identifiées sont ensuite analysées par régression OLS afin d’évaluer leur effet sur la performance des hôtes et leur interaction avec le statut de Superhôte.

## 5 tactiques adaptées au contexte d’Airbnb:
- tactiques relationnelles : Ouveture, sociabilité, authenticité
- tactiques promotionnelles : Auto-promotion, exemplarité

## Jeu de données 
- Source : [Inside Airbnb](https://insideairbnb.com/get-the-data/)  
- Échantillon : logements Airbnb à Paris en décembre 2023
- Taille : ~ 75 000 annonces
- Variable clé : description personnelle de l'hôte (*host_about*)

## Méthodologie 
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
- Superhôte vs. Autres : OLS avec terme d'intéraction, t-test, Cohen's d

## 📊 Résultats principaux
- L’**auto-promotion** et l’**exemplarité** sont les plus courantes;
- La **sociabilité** et l’**authenticité** élève le taux de réservation, alors que l’**exemplarité** est contre-productif, d'autres ne sont pas significatives; 
- Le statut de **Superhôte** joue un rôle modérateur, il active l'effet négatif de l'**auto-promotion** , mais peut atténuer l'effet négatif de l'**exemplarité**. D'autres tactiques restent constantes.

## Structure du projet



## Limites 



