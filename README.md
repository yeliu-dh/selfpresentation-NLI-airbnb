> Languages : 🇫🇷 [Français](#version-française) · 🇬🇧 [English](#english-version) · 🇨🇳 [中文](#中文版介绍) 

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
### Pipeline         
        
            Définition du construit                        
                        │                           
                Analyse qualitative    
                        │                        
            ┌─> Echantillon des          
            |   items descriptifs    
            |           │                   
            |     Traitement NLI
            |  Classification zéro-shot           
            |   (bge-m3-zeroshot-v2.0)       
  Révision  |           │                        
  des items |   Analyse factorielle 
            |    exploratoire (EFA)                 
            |           |                   
            | Validation psychométrique 
            |   (Chronbah'α, commnalité,
            |        complexité)           
            |           |                   
            └────────── |
                        |                   
                Echelle validée                                      
                        |                   
                Scores pondérés 
                des tactiques 


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

## 🗂️ Structure du projet
- **`scripts/`** : fichier `.ipynb` pour le prétraitement, l'analyse factorielle et la modélisation
- **`utils/`** : fonctions utilitaires
- **`*_results`** : résultats des traitements et des analyses

## ⚙️ Reproduction
- **Python** : `3.10`
- **Dépendances** : `pip install -r requirements.txt`
- **Données** : non incluses ; scripts de reproduction fournis. Disponibles sur demande.


---
# English Version:
# Self-Presentation and Host Performance on Airbnb: A Psychometric Approach Using Zero-Shot NLI

## 🔍 Abstract

This research examines the impact of Airbnb hosts’ self-presentation tactics on their booking rate and the moderating role of Superhost status, based on a 22-item psychometric scale applied at scale through zero-shot NLI.

## 🏷️ 5 Tactics Adapted to the Airbnb Context:

* **relational tactics**: Openness, sociability, authenticity

* **promotional tactics**: Self-promotion, exemplification

## 📊 Dataset

* Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/)

* Sample: Airbnb listings in Paris in December 2023

* Size: ~ 75,000 listings

* Key variable: host’s personal description (*host_about*)

## 🔬 Methodology


### Scale Construction

* EFA (Exploratory Factor Analysis)

* Psychometric validation: Cronbach’s α, communalities, complexity

* Validated scale: 5 tactics → 22 descriptive items

### Natural Language Inference (NLI)

* Multilingual zero-shot classification: evaluating the extent to which each item is semantically present in a given text

* Model: [bge-m3-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0)

* Weighted scores: 22 item scores → 5 dimension scores

### Econometric Analysis

* Tactics & monthly booking rate: OLS regression

* Superhost vs. Others: OLS with interaction term, t-test, Cohen’s *d*

## 💡 Main Results

* **Prevalence**: Self-promotion and exemplarity are the most common;

* **Impact**: Sociability and authenticity increase the booking rate, while exemplarity has a negative effect; the others show no significant effect;

* **Moderation**: Superhost status plays a moderating role: it reveals the negative effect of self-promotion, but may attenuate the negative effect of exemplarity. The effects of the other tactics remain unchanged.

## 🗂️ Project Structure

* **`scripts/`**: `.ipynb` file for preprocessing, factor analysis, and modeling

* **`utils/`**: utility functions

* **`*_results`**: results of the processing and analyses

## ⚙️ Reproduction

* **Python**: `3.10`

* **Dependencies**: `pip install -r requirements.txt`

* **Data**: not included; reproduction scripts provided. Available upon request.



# 中文版介绍
# 爱彼迎房东自我展示策略与预定绩效：基于多维心理测量和零样本分类的计算分析

## 🔍 摘要

本研究基于一套包含22个条目的心理测量量表，并通过零样本NLI进行大规模应用，研究Airbnb房东的自我呈现策略对其预订率的影响，以及超级房东勋章的调节作用。

## 🏷️ 适用于Airbnb情境的5种策略：

* **关系型策略**：开放性、社交性、真实性

* **促销型策略**：自我推销、榜样化

## 📊 数据集

* 数据来源：[Inside Airbnb](https://insideairbnb.com/get-the-data/)

* 样本：2023年12月巴黎的Airbnb房源

* 规模：~ 75,000条房源

* 关键变量：房东个人描述（*host_about*）

## 🔬 研究方法

### 量表构建

* EFA（探索性因子分析）

* 心理测量学验证：Cronbach’s α、共同度、复杂度

* 验证后的量表：5种策略 → 22个描述性条目

### 自然语言推理（NLI）

* 多语言零样本分类：评估每个条目在给定文本中以语义方式呈现的程度

* 大语言模型：[bge-m3-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0)

* 加权得分：22个条目得分 → 5个维度得分

### 计量经济学分析

* 策略与月度预订率的相关度：OLS回归

* 超级房东 vs. 其他房东：带交互项的OLS、t检验、Cohen’s *d*

## 💡 主要研究结果

* **普遍性**：自我推销和示范性是最常见的策略；

* **影响**：社交性和真实性提高预订率，而示范性具有负向影响；其他策略没有显著影响；

* **调节作用**：Superhost身份发挥调节作用：它揭示了自我推销的负向影响，但可能减弱示范性的负向影响。其他策略的影响保持不变。

## 🗂️ 项目结构

* **`scripts/`**：用于数据预处理、因子分析和建模的`.ipynb`文件

* **`utils/`**：工具函数

* **`*_results`**：处理和分析结果

## ⚙️ 复现

* **Python**：`3.10`

* **依赖**：`pip install -r requirements.txt`

* **数据**：仓库未包含原始数据；提供复现脚本，按需提供数据。
