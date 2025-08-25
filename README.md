# Credit Scoring - Projet End-to-End

Projet complet de scoring de crédit utilisant le machine learning, de la préparation des données à la mise en production.

## 🏗️ Architecture

```
credit-scoring/
├─ data/                  # Données brutes et traitées
├─ notebooks/             # Jupyter notebooks pour l'analyse
├─ src/                   # Code source Python
├─ api/                   # API FastAPI pour le scoring
├─ app/                   # Interface Streamlit
├─ models/                # Modèles entraînés
├─ reports/               # Rapports et visualisations
└─ tests/                 # Tests unitaires
```

## 🚀 Installation

```bash
# Cloner le repository
git clone <repository-url>
cd credit-scoring

# Installer les dépendances
make install

# Ou avec pip
pip install -e .
```

## 📊 Utilisation

### 1. Préparation des données
```bash
make data
```

### 2. Création des features
```bash
make features
```

### 3. Entraînement des modèles
```bash
make train
```

### 4. Évaluation
```bash
make evaluate
```

### 5. Lancement de l'API
```bash
make api
```

### 6. Lancement de l'interface
```bash
make app
```

## 🔧 Modèles disponibles

- **Logistic Regression** : Modèle de base avec interprétabilité
- **Scorecard** : Score de 0 à 1000 avec vingtiles
- **Classes de Risque** : Classification automatique (KMeans/CAH)
- **XGBoost** : Modèle challenger avec SHAP

## 📈 Métriques d'évaluation

- AUC-ROC
- Gini
- F1-Score
- Kolmogorov-Smirnov (KS)
- Courbes ROC et PR

## 🧪 Tests

```bash
# Tests unitaires
make test

# Tests avec couverture
make test-cov
```

## 📝 Commandes utiles

```bash
# Voir toutes les commandes disponibles
make help

# Formater le code
make format

# Vérifier la qualité du code
make lint

# Nettoyer les fichiers temporaires
make clean
```

## 🎯 Endpoints API

- `GET /health` : Vérification de l'état de l'API
- `POST /score` : Calcul du score de crédit
- `POST /chr` : Classification en classes de risque

## 📱 Interface Streamlit

Interface utilisateur simple pour :
- Upload de fichiers CSV
- Saisie manuelle de données
- Affichage des scores et classes de risque
- Visualisations interactives

## 🔒 Sécurité

- Validation des données d'entrée
- Gestion des erreurs robuste
- Logging des requêtes
- Tests de sécurité

## 📚 Dépendances principales

- **Data Science** : pandas, numpy, scikit-learn
- **ML** : xgboost, shap
- **Visualisation** : matplotlib, seaborn, plotly
- **API** : FastAPI, uvicorn
- **Interface** : Streamlit
- **Tests** : pytest

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commit les changements
4. Push vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT.



