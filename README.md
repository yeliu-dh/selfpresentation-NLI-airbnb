#  Présentation de soi et performance des hôtes sur Airbnb: une approche psychométrique par classification zéro-shot
 
## 🧠 Résumé
Basé sur la théorie du managment d'impression de Goffman, ce projet explore comment les hôtes d’Airbnb se présentent dans leurs descriptions personnelles, à l'aide d'un modèle de zéro-shot classification et examine l'enjeu socio-économique des tactiques via les modèles de régression OLS.

## 🏷️ Mots-clés
Humanités numériques · Goffman · Présentation de soi · Managment d'impression · Airbnb · Zero-shot Classification · Psychométrie ·  Traitement automatique du language naturel · Analyse sémantique

## 💡 5 tactiques identifiées :
### personnalité :
- **Ouverture** : 'open to different cultures', 'cosmopolitan','international view', 'cultural exchange'
- **Authenticité** : 'personal life', 'life experiences', 'divers interests', 'hobbies', 'enjoy life'
- **Sociabilité**: 'meet new people', 'welcoming', 'friendly', 'sociable', 'interpersonal interaction'


### marketing :
- **Auto-promotion** : 'thoughtful service', 'attentive to needs', 'willing to help', 'responsive'
- **Exemplarité** : 'fan of Airbnb', 'Airbnb community','love Airbnb', 'travel with Airbnb'
  

## 📊 Résultats principaux
- L’**auto-promotion** et l’**exemplarité** sont les plus courantes;
- La **sociabilité** et l’**authenticité** élève le taux de réservation, alors que l’**exemplarité** est contre-productif, d'autres ne sont pas significatives; 
- Le statut de **Superhôte** joue un rôle modérateur, il active l'effet négatif de l'**auto-promotion** , mais peut atténuer l'effet négatif de l'**exemplarité**. D'autres tactiques restent constantes.


## 📁 Structure du projet
- `code/` — Scripts principaux d’analyse et de traitement
-  `corpus/` — Données brutes issues des listings Airbnb
- `data/` — Données traitées en fonction de langue et la version qui inclut les scores de zéro-shot classification
- `figs/` — Visualisations générées à partir des analyses
- `result_models/` — Données combinées, inclus les scores de zsc et les scores fusionnées en fonction de tactiques 

