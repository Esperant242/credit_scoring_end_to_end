"""
Configuration du projet de scoring de crédit
"""
import os
from pathlib import Path
import yaml
from typing import Dict, List, Any

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Créer les dossiers s'ils n'existent pas
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Paramètres par défaut
DEFAULT_CONFIG = {
    "random_state": 42,
    "test_size": 0.15,
    "validation_size": 0.15,
    "target_column": "default_status",
    "positive_class": 1,
    "negative_class": 0
}

def load_config(config_file: str = "features.yaml") -> Dict[str, Any]:
    """Charge la configuration depuis un fichier YAML"""
    config_path = CONFIGS_DIR / config_file
    
    if not config_path.exists():
        print(f"Fichier de configuration {config_path} non trouvé, utilisation des valeurs par défaut")
        return DEFAULT_CONFIG
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Erreur lors du chargement de la configuration: {e}")
        return DEFAULT_CONFIG

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return load_config("features.yaml")

def get_numerical_features() -> List[str]:
    """Retourne la liste des features numériques"""
    config = get_feature_config()
    return config.get("numerical_features", []) + config.get("discrete_features", [])

def get_categorical_features() -> List[str]:
    """Retourne la liste des features catégorielles"""
    config = get_feature_config()
    return config.get("categorical_features", [])

def get_target_column() -> str:
    """Retourne le nom de la colonne cible"""
    config = get_feature_config()
    return config.get("target_column", "default_status")

def get_preprocessing_config() -> Dict[str, Any]:
    """Retourne la configuration du preprocessing"""
    config = get_feature_config()
    return config.get("preprocessing", {})

# Chemins des fichiers de données
def get_data_paths() -> Dict[str, Path]:
    """Retourne les chemins des fichiers de données"""
    return {
        "raw": RAW_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
        "train": PROCESSED_DATA_DIR / "train.parquet",
        "validation": PROCESSED_DATA_DIR / "validation.parquet",
        "test": PROCESSED_DATA_DIR / "test.parquet"
    }

# Chemins des modèles
def get_model_paths() -> Dict[str, Path]:
    """Retourne les chemins des modèles"""
    return {
        "preprocessor": MODELS_DIR / "preprocessor.pkl",
        "logistic": MODELS_DIR / "logistic_model.pkl",
        "scorecard": MODELS_DIR / "scorecard.json",
        "chr": MODELS_DIR / "chr_model.pkl",
        "xgb": MODELS_DIR / "xgb_model.pkl"
    }

# Configuration des métriques
METRICS_CONFIG = {
    "classification": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "regression": ["mse", "mae", "r2"],
    "ranking": ["ndcg", "map"]
}

# Configuration des plots
PLOT_CONFIG = {
    "figsize": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8",
    "palette": "viridis"
}

if __name__ == "__main__":
    # Test de la configuration
    print("Configuration chargée avec succès!")
    print(f"Features numériques: {get_numerical_features()}")
    print(f"Features catégorielles: {get_categorical_features()}")
    print(f"Colonne cible: {get_target_column()}")



