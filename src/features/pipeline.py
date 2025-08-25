"""
Pipeline de features pour le scoring de crédit
Imputation, gestion des outliers, encodage et scaling
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import des modules locaux
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import get_data_paths, get_feature_config, get_model_paths
from src.utils.logging import setup_logger, log_execution_time
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Configuration du logger
logger = setup_logger("feature_pipeline")

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Gestionnaire d'outliers basé sur la méthode IQR"""
    
    def __init__(self, method='iqr', threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
    
    def fit(self, X, y=None):
        """Calcule les bornes pour chaque feature numérique"""
        if isinstance(X, pd.DataFrame):
            numerical_cols = X.select_dtypes(include=[np.number]).columns
        else:
            numerical_cols = range(X.shape[1])
        
        for col in numerical_cols:
            if self.method == 'iqr':
                Q1 = np.percentile(X[col], 25)
                Q3 = np.percentile(X[col], 75)
                IQR = Q3 - Q1
                self.lower_bounds_[col] = Q1 - self.threshold * IQR
                self.upper_bounds_[col] = Q3 + self.threshold * IQR
            elif self.method == 'zscore':
                mean = np.mean(X[col])
                std = np.std(X[col])
                self.lower_bounds_[col] = mean - self.threshold * std
                self.upper_bounds_[col] = mean + self.threshold * std
        
        return self
    
    def transform(self, X):
        """Applique le capping des outliers"""
        X_transformed = X.copy()
        
        if isinstance(X, pd.DataFrame):
            for col in self.lower_bounds_.keys():
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].clip(
                        lower=self.lower_bounds_[col],
                        upper=self.upper_bounds_[col]
                    )
        else:
            for col in self.lower_bounds_.keys():
                X_transformed[:, col] = np.clip(
                    X_transformed[:, col],
                    self.lower_bounds_[col],
                    self.upper_bounds_[col]
                )
        
        return X_transformed

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encodeur cible pour les variables catégorielles"""
    
    def __init__(self, min_samples_leaf=1, smoothing=1.0):
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.target_means_ = {}
        self.global_mean_ = None
    
    def fit(self, X, y):
        """Calcule les moyennes cibles pour chaque catégorie"""
        self.global_mean_ = np.mean(y)
        
        for col in X.columns:
            if X[col].dtype == 'object':
                agg = X.groupby(col).agg({
                    'target': ['count', 'mean']
                }).reset_index()
                
                agg.columns = ['category', 'count', 'mean']
                
                # Lissage
                smoothed_mean = (
                    (agg['count'] * agg['mean'] + self.smoothing * self.global_mean_) /
                    (agg['count'] + self.smoothing)
                )
                
                self.target_means_[col] = dict(zip(agg['category'], smoothed_mean))
        
        return self
    
    def transform(self, X):
        """Transforme les variables catégorielles en moyennes cibles"""
        X_transformed = X.copy()
        
        for col in self.target_means_.keys():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].map(
                    self.target_means_[col]
                ).fillna(self.global_mean_)
        
        return X_transformed

class FeaturePipeline:
    """Pipeline complet de préparation des features"""
    
    def __init__(self):
        self.config = get_feature_config()
        self.data_paths = get_data_paths()
        self.model_paths = get_model_paths()
        self.preprocessor = None
        self.feature_names = None
        
    @log_execution_time
    def create_preprocessing_pipeline(self) -> Pipeline:
        """Crée le pipeline de preprocessing"""
        logger.info("Création du pipeline de preprocessing...")
        
        # Configuration
        preprocessing_config = self.config.get('preprocessing', {})
        numerical_features = self.config.get('numerical_features', [])
        discrete_features = self.config.get('discrete_features', [])
        categorical_features = self.config.get('categorical_features', [])
        
        # Features numériques (continues + discrètes)
        all_numerical = numerical_features + discrete_features
        
        # Pipeline pour les features numériques
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=preprocessing_config.get('numerical_imputation', 'median'))),
            ('outlier_handler', OutlierHandler(
                method=preprocessing_config.get('outlier_method', 'iqr'),
                threshold=preprocessing_config.get('outlier_threshold', 1.5)
            )),
            ('scaler', RobustScaler())
        ])
        
        # Pipeline pour les features catégorielles
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=preprocessing_config.get('categorical_imputation', 'mode'))),
            ('target_encoder', TargetEncoder())
        ])
        
        # Transformer de colonnes
        preprocessors = [
            ('numerical', numerical_pipeline, all_numerical),
            ('categorical', categorical_pipeline, categorical_features)
        ]
        
        self.preprocessor = ColumnTransformer(
            preprocessors,
            remainder='drop'
        )
        
        logger.info("✅ Pipeline de preprocessing créé")
        return self.preprocessor
    
    @log_execution_time
    def fit_preprocessor(self, train_data: pd.DataFrame) -> None:
        """Entraîne le préprocesseur sur les données d'entraînement"""
        logger.info("Entraînement du préprocesseur...")
        
        if self.preprocessor is None:
            self.create_preprocessing_pipeline()
        
        # Préparer les données pour l'entraînement
        X_train = train_data.drop('default_status', axis=1)
        y_train = train_data['default_status']
        
        # Ajouter la colonne cible pour l'encodeur cible
        X_train_with_target = X_train.copy()
        X_train_with_target['target'] = y_train
        
        # Entraîner le préprocesseur
        self.preprocessor.fit(X_train_with_target, y_train)
        
        # Sauvegarder les noms des features
        self.feature_names = self.preprocessor.get_feature_names_out()
        
        logger.info(f"✅ Préprocesseur entraîné avec {len(self.feature_names)} features")
    
    @log_execution_time
    def transform_data(self, data: pd.DataFrame) -> np.ndarray:
        """Transforme les données avec le préprocesseur entraîné"""
        if self.preprocessor is None:
            raise ValueError("Le préprocesseur doit être entraîné avant la transformation")
        
        # Préparer les données
        if 'default_status' in data.columns:
            X = data.drop('default_status', axis=1)
            # Ajouter une colonne cible factice pour l'encodeur cible
            X['target'] = 0  # Valeur par défaut
        else:
            X = data
            X['target'] = 0
        
        # Transformer
        X_transformed = self.preprocessor.transform(X)
        
        return X_transformed
    
    @log_execution_time
    def save_preprocessor(self) -> None:
        """Sauvegarde le préprocesseur entraîné"""
        if self.preprocessor is None:
            logger.warning("Aucun préprocesseur à sauvegarder")
            return
        
        try:
            preprocessor_path = self.model_paths['preprocessor']
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            
            logger.info(f"✅ Préprocesseur sauvegardé: {preprocessor_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
    
    @log_execution_time
    def load_preprocessor(self) -> None:
        """Charge le préprocesseur sauvegardé"""
        try:
            preprocessor_path = self.model_paths['preprocessor']
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            self.feature_names = self.preprocessor.get_feature_names_out()
            logger.info(f"✅ Préprocesseur chargé: {preprocessor_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
    
    @log_execution_time
    def run_pipeline(self) -> dict:
        """Exécute le pipeline complet de features"""
        logger.info("🚀 Démarrage du pipeline de features")
        
        try:
            # 1. Charger les données d'entraînement
            logger.info("Chargement des données d'entraînement...")
            train_data = pd.read_parquet(self.data_paths['train'])
            logger.info(f"Données d'entraînement chargées: {train_data.shape}")
            
            # 2. Créer et entraîner le préprocesseur
            self.fit_preprocessor(train_data)
            
            # 3. Transformer les données d'entraînement
            logger.info("Transformation des données d'entraînement...")
            X_train_transformed = self.transform_data(train_data)
            logger.info(f"Features transformées: {X_train_transformed.shape}")
            
            # 4. Sauvegarder le préprocesseur
            self.save_preprocessor()
            
            # 5. Transformer les autres datasets
            logger.info("Transformation des données de validation et test...")
            
            validation_data = pd.read_parquet(self.data_paths['validation'])
            test_data = pd.read_parquet(self.data_paths['test'])
            
            X_val_transformed = self.transform_data(validation_data)
            X_test_transformed = self.transform_data(test_data)
            
            # 6. Sauvegarder les features transformées
            logger.info("Sauvegarde des features transformées...")
            
            # Créer des DataFrames avec les features transformées
            feature_names = self.feature_names
            
            train_features_df = pd.DataFrame(
                X_train_transformed, 
                columns=feature_names,
                index=train_data.index
            )
            train_features_df['default_status'] = train_data['default_status']
            
            val_features_df = pd.DataFrame(
                X_val_transformed, 
                columns=feature_names,
                index=validation_data.index
            )
            val_features_df['default_status'] = validation_data['default_status']
            
            test_features_df = pd.DataFrame(
                X_test_transformed, 
                columns=feature_names,
                index=test_data.index
            )
            test_features_df['default_status'] = test_data['default_status']
            
            # Sauvegarder
            train_features_df.to_parquet(self.data_paths['processed'] / 'train_features.parquet')
            val_features_df.to_parquet(self.data_paths['processed'] / 'validation_features.parquet')
            test_features_df.to_parquet(self.data_paths['processed'] / 'test_features.parquet')
            
            # 7. Résumé final
            pipeline_summary = {
                "status": "success",
                "train_features_shape": X_train_transformed.shape,
                "validation_features_shape": X_val_transformed.shape,
                "test_features_shape": X_test_transformed.shape,
                "feature_names": list(feature_names),
                "preprocessor_saved": True
            }
            
            logger.info("✅ Pipeline de features terminé avec succès")
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline de features: {e}")
            raise

def main():
    """Fonction principale"""
    pipeline = FeaturePipeline()
    
    try:
        results = pipeline.run_pipeline()
        
        print("\n" + "="*50)
        print("RÉSUMÉ DU PIPELINE DE FEATURES")
        print("="*50)
        print(f"Features d'entraînement: {results['train_features_shape']}")
        print(f"Features de validation: {results['validation_features_shape']}")
        print(f"Features de test: {results['test_features_shape']}")
        print(f"Nombre de features: {len(results['feature_names'])}")
        print(f"Préprocesseur sauvegardé: {'✅' if results['preprocessor_saved'] else '❌'}")
        
        print(f"\nPremières features:")
        for i, feature in enumerate(results['feature_names'][:10]):
            print(f"  {i+1:2d}. {feature}")
        
        if len(results['feature_names']) > 10:
            print(f"  ... et {len(results['feature_names']) - 10} autres")
        
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Erreur dans le pipeline principal: {e}")
        raise

if __name__ == "__main__":
    main()



