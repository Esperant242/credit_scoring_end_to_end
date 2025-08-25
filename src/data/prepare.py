"""
Préparation des données pour le scoring de crédit
Lecture, EDA rapide et split stratifié 70/15/15
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import des modules locaux
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import get_data_paths, get_feature_config
from src.utils.logging import setup_logger, log_execution_time
from src.data.schema import CreditDataValidator, check_missing_columns

# Configuration du logger
logger = setup_logger("data_preparation")

class CreditDataPreparer:
    """Classe pour préparer les données de crédit"""
    
    def __init__(self):
        self.config = get_feature_config()
        self.data_paths = get_data_paths()
        self.validator = CreditDataValidator()
        self.raw_data = None
        self.processed_data = None
        
    @log_execution_time
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV
        
        Args:
            file_path: Chemin vers le fichier CSV
            
        Returns:
            DataFrame chargé
        """
        if file_path is None:
            # Chercher un fichier CSV dans le dossier raw
            raw_dir = self.data_paths["raw"]
            csv_files = list(raw_dir.glob("*.csv"))
            
            if not csv_files:
                logger.warning("Aucun fichier CSV trouvé dans le dossier raw, création de données synthétiques")
                return self._create_synthetic_data()
            
            file_path = csv_files[0]
            logger.info(f"Fichier trouvé: {file_path}")
        
        try:
            logger.info(f"Chargement des données depuis {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Données chargées: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            logger.info("Création de données synthétiques")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Crée des données synthétiques pour le développement
        
        Args:
            n_samples: Nombre d'échantillons à créer
            
        Returns:
            DataFrame avec des données synthétiques
        """
        logger.info(f"Création de {n_samples} échantillons synthétiques")
        
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(35, 10, n_samples).clip(18, 70),
            'income': np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000),
            'debt_ratio': np.random.beta(2, 5, n_samples) * 2,
            'monthly_income': np.random.lognormal(8.5, 0.4, n_samples).clip(1500, 15000),
            'credit_utilization': np.random.beta(2, 3, n_samples),
            'payment_history_length': np.random.exponential(3, n_samples).clip(0, 20),
            'number_of_credit_cards': np.random.poisson(3, n_samples).clip(0, 10),
            'loan_amount': np.random.lognormal(9.5, 0.6, n_samples).clip(5000, 100000),
            'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
            'number_of_dependents': np.random.poisson(1.5, n_samples).clip(0, 5),
            'number_of_open_accounts': np.random.poisson(5, n_samples).clip(0, 20),
            'number_of_credit_inquiries': np.random.poisson(2, n_samples).clip(0, 10),
            'months_employed': np.random.exponential(60, n_samples).clip(0, 300),
            'employment_status': np.random.choice(['employed', 'unemployed', 'self-employed', 'retired'], n_samples, p=[0.7, 0.1, 0.15, 0.05]),
            'education_level': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples, p=[0.3, 0.4, 0.25, 0.05]),
            'marital_status': np.random.choice(['single', 'married', 'divorced', 'widowed'], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
            'home_ownership': np.random.choice(['own', 'rent', 'mortgage', 'other'], n_samples, p=[0.3, 0.3, 0.35, 0.05]),
            'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'medical', 'other'], n_samples, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
            'credit_history_length': np.random.choice(['new', 'short', 'medium', 'long', 'excellent'], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Créer la variable cible (défaut) basée sur des règles métier
        default_prob = (
            (df['debt_ratio'] > 0.5).astype(int) * 0.3 +
            (df['credit_utilization'] > 0.8).astype(int) * 0.2 +
            (df['number_of_credit_inquiries'] > 5).astype(int) * 0.15 +
            (df['age'] < 25).astype(int) * 0.1 +
            (df['income'] < 30000).astype(int) * 0.1 +
            (df['employment_status'] == 'unemployed').astype(int) * 0.15
        )
        
        df['default_status'] = np.random.binomial(1, default_prob.clip(0, 0.8))
        
        logger.info(f"Données synthétiques créées: {df.shape}")
        return df
    
    @log_execution_time
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Valide les données selon le schéma
        
        Args:
            df: DataFrame à valider
            
        Returns:
            True si valide, False sinon
        """
        logger.info("Validation des données...")
        
        # Vérifier les colonnes manquantes
        missing_info = check_missing_columns(df)
        logger.info(f"Colonnes manquantes: {missing_info['missing']}")
        logger.info(f"Colonnes supplémentaires: {missing_info['extra']}")
        
        # Valider avec le schéma
        is_valid, errors = self.validator.validate_dataframe(df)
        
        if not is_valid:
            logger.warning(f"Données invalides: {len(errors)} erreurs")
            for error in errors[:5]:  # Afficher les 5 premières erreurs
                logger.warning(error)
        else:
            logger.info("✅ Données validées avec succès")
        
        return is_valid
    
    @log_execution_time
    def perform_eda(self, df: pd.DataFrame) -> dict:
        """
        Effectue une analyse exploratoire rapide des données
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            Dictionnaire avec les résultats de l'EDA
        """
        logger.info("Analyse exploratoire des données...")
        
        eda_results = {}
        
        # Informations de base
        eda_results['shape'] = df.shape
        eda_results['dtypes'] = df.dtypes.to_dict()
        eda_results['memory_usage'] = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Valeurs manquantes
        missing_values = df.isnull().sum()
        eda_results['missing_values'] = missing_values[missing_values > 0].to_dict()
        eda_results['missing_percentage'] = (missing_values / len(df) * 100).round(2).to_dict()
        
        # Statistiques descriptives
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        eda_results['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        # Distribution de la cible
        if 'default_status' in df.columns:
            target_dist = df['default_status'].value_counts(normalize=True)
            eda_results['target_distribution'] = target_dist.to_dict()
            eda_results['class_imbalance'] = target_dist.max() / target_dist.min()
        
        # Corrélations
        if len(numerical_cols) > 1:
            correlation_matrix = df[numerical_cols].corr()
            eda_results['correlations'] = correlation_matrix.to_dict()
        
        logger.info("✅ EDA terminée")
        return eda_results
    
    @log_execution_time
    def split_data(self, df: pd.DataFrame, test_size: float = 0.15, 
                   validation_size: float = 0.15, random_state: int = 42) -> tuple:
        """
        Divise les données en train/validation/test de manière stratifiée
        
        Args:
            df: DataFrame à diviser
            test_size: Proportion pour le test
            validation_size: Proportion pour la validation
            random_state: Seed pour la reproductibilité
            
        Returns:
            Tuple (train, validation, test)
        """
        logger.info("Division des données en train/validation/test...")
        
        from sklearn.model_selection import train_test_split
        
        # Séparer d'abord train+validation et test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['default_status'] if 'default_status' in df.columns else None
        )
        
        # Puis séparer train et validation
        val_size_adjusted = validation_size / (1 - test_size)
        train, validation = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val['default_status'] if 'default_status' in train_val.columns else None
        )
        
        logger.info(f"Train: {train.shape}, Validation: {validation.shape}, Test: {test.shape}")
        
        return train, validation, test
    
    @log_execution_time
    def save_splits(self, train: pd.DataFrame, validation: pd.DataFrame, 
                    test: pd.DataFrame) -> None:
        """
        Sauvegarde les splits en format parquet
        
        Args:
            train: Données d'entraînement
            validation: Données de validation
            test: Données de test
        """
        logger.info("Sauvegarde des splits...")
        
        try:
            train.to_parquet(self.data_paths["train"], index=False)
            validation.to_parquet(self.data_paths["validation"], index=False)
            test.to_parquet(self.data_paths["test"], index=False)
            
            logger.info("✅ Splits sauvegardés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
    
    @log_execution_time
    def run_pipeline(self, file_path: str = None) -> dict:
        """
        Exécute le pipeline complet de préparation des données
        
        Args:
            file_path: Chemin vers le fichier CSV (optionnel)
            
        Returns:
            Résultats du pipeline
        """
        logger.info("🚀 Démarrage du pipeline de préparation des données")
        
        # 1. Charger les données
        df = self.load_data(file_path)
        self.raw_data = df
        
        # 2. Valider les données
        is_valid = self.validate_data(df)
        if not is_valid:
            logger.warning("Données invalides détectées, continuation avec validation partielle")
        
        # 3. EDA rapide
        eda_results = self.perform_eda(df)
        
        # 4. Diviser les données
        train, validation, test = self.split_data(df)
        
        # 5. Sauvegarder les splits
        self.save_splits(train, validation, test)
        
        # 6. Résumé final
        pipeline_summary = {
            "status": "success",
            "raw_data_shape": df.shape,
            "train_shape": train.shape,
            "validation_shape": validation.shape,
            "test_shape": test.shape,
            "eda_results": eda_results,
            "validation_status": is_valid
        }
        
        logger.info("✅ Pipeline de préparation terminé avec succès")
        return pipeline_summary

def main():
    """Fonction principale"""
    preparer = CreditDataPreparer()
    
    try:
        results = preparer.run_pipeline()
        
        print("\n" + "="*50)
        print("RÉSUMÉ DE LA PRÉPARATION DES DONNÉES")
        print("="*50)
        print(f"Données brutes: {results['raw_data_shape']}")
        print(f"Train: {results['train_shape']}")
        print(f"Validation: {results['validation_shape']}")
        print(f"Test: {results['test_shape']}")
        print(f"Statut validation: {'✅' if results['validation_status'] else '⚠️'}")
        
        if 'target_distribution' in results['eda_results']:
            print(f"\nDistribution de la cible:")
            for class_name, proportion in results['eda_results']['target_distribution'].items():
                print(f"  Classe {class_name}: {proportion:.3f}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Erreur dans le pipeline principal: {e}")
        raise

if __name__ == "__main__":
    main()



