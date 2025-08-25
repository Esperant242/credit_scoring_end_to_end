"""
Tests pour la préparation des données
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from src.data.prepare import CreditDataPreparer
from src.data.schema import CreditDataValidator, CreditDataSchema

class TestCreditDataPreparer:
    """Tests pour la classe CreditDataPreparer"""
    
    def setup_method(self):
        """Setup avant chaque test"""
        self.preparer = CreditDataPreparer()
    
    def test_create_synthetic_data(self):
        """Test de création de données synthétiques"""
        df = self.preparer._create_synthetic_data(100)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert 'default_status' in df.columns
        
        # Vérifier les colonnes requises
        required_cols = [
            'age', 'income', 'debt_ratio', 'monthly_income', 'credit_utilization',
            'payment_history_length', 'number_of_credit_cards', 'loan_amount', 'loan_term',
            'number_of_dependents', 'number_of_open_accounts', 'number_of_credit_inquiries',
            'months_employed', 'employment_status', 'education_level', 'marital_status',
            'home_ownership', 'loan_purpose', 'credit_history_length'
        ]
        
        for col in required_cols:
            assert col in df.columns
    
    def test_validate_data(self):
        """Test de validation des données"""
        # Créer des données valides
        df = self.preparer._create_synthetic_data(50)
        
        # Valider
        is_valid = self.preparer.validate_data(df)
        
        # Les données synthétiques devraient être valides
        assert is_valid == True
    
    def test_perform_eda(self):
        """Test de l'analyse exploratoire"""
        df = self.preparer._create_synthetic_data(100)
        
        eda_results = self.preparer.perform_eda(df)
        
        assert isinstance(eda_results, dict)
        assert 'shape' in eda_results
        assert 'target_distribution' in eda_results
        assert eda_results['shape'] == (100, 19)  # 18 features + 1 cible
    
    def test_split_data(self):
        """Test de division des données"""
        df = self.preparer._create_synthetic_data(1000)
        
        train, validation, test = self.preparer.split_data(df)
        
        # Vérifier les tailles (70/15/15)
        assert len(train) == 700
        assert len(validation) == 150
        assert len(test) == 150
        
        # Vérifier que la somme fait 1000
        assert len(train) + len(validation) + len(test) == 1000
        
        # Vérifier que toutes les lignes sont présentes
        all_indices = set(train.index) | set(validation.index) | set(test.index)
        assert len(all_indices) == 1000

class TestCreditDataValidator:
    """Tests pour la classe CreditDataValidator"""
    
    def setup_method(self):
        """Setup avant chaque test"""
        self.validator = CreditDataValidator()
    
    def test_validate_valid_data(self):
        """Test de validation de données valides"""
        valid_data = {
            'age': 30.0,
            'income': 50000.0,
            'debt_ratio': 0.3,
            'monthly_income': 4166.67,
            'credit_utilization': 0.2,
            'payment_history_length': 5.0,
            'number_of_credit_cards': 2,
            'loan_amount': 15000.0,
            'loan_term': 36,
            'number_of_dependents': 1,
            'number_of_open_accounts': 3,
            'number_of_credit_inquiries': 1,
            'months_employed': 24,
            'employment_status': 'employed',
            'education_level': 'bachelor',
            'marital_status': 'single',
            'home_ownership': 'rent',
            'loan_purpose': 'debt_consolidation',
            'credit_history_length': 'medium'
        }
        
        df = pd.DataFrame([valid_data])
        is_valid, errors = self.validator.validate_dataframe(df)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_invalid_data(self):
        """Test de validation de données invalides"""
        invalid_data = {
            'age': 150.0,  # Âge invalide
            'income': 50000.0,
            'debt_ratio': 0.3,
            'monthly_income': 4166.67,
            'credit_utilization': 0.2,
            'payment_history_length': 5.0,
            'number_of_credit_cards': 2,
            'loan_amount': 15000.0,
            'loan_term': 36,
            'number_of_dependents': 1,
            'number_of_open_accounts': 3,
            'number_of_credit_inquiries': 1,
            'months_employed': 24,
            'employment_status': 'employed',
            'education_level': 'bachelor',
            'marital_status': 'single',
            'home_ownership': 'rent',
            'loan_purpose': 'debt_consolidation',
            'credit_history_length': 'medium'
        }
        
        df = pd.DataFrame([invalid_data])
        is_valid, errors = self.validator.validate_dataframe(df)
        
        assert is_valid == False
        assert len(errors) > 0

class TestCreditDataSchema:
    """Tests pour le schéma Pydantic"""
    
    def test_valid_schema(self):
        """Test de validation du schéma avec des données valides"""
        valid_data = {
            'age': 30.0,
            'income': 50000.0,
            'debt_ratio': 0.3,
            'monthly_income': 4166.67,
            'credit_utilization': 0.2,
            'payment_history_length': 5.0,
            'number_of_credit_cards': 2,
            'loan_amount': 15000.0,
            'loan_term': 36,
            'number_of_dependents': 1,
            'number_of_open_accounts': 3,
            'number_of_credit_inquiries': 1,
            'months_employed': 24,
            'employment_status': 'employed',
            'education_level': 'bachelor',
            'marital_status': 'single',
            'home_ownership': 'rent',
            'loan_purpose': 'debt_consolidation',
            'credit_history_length': 'medium'
        }
        
        # Créer l'objet schema
        schema = CreditDataSchema(**valid_data)
        
        # Vérifier que les données sont correctes
        assert schema.age == 30.0
        assert schema.income == 50000.0
        assert schema.employment_status == 'employed'
    
    def test_invalid_schema(self):
        """Test de validation du schéma avec des données invalides"""
        invalid_data = {
            'age': 150.0,  # Âge invalide
            'income': 50000.0,
            'debt_ratio': 0.3,
            'monthly_income': 4166.67,
            'credit_utilization': 0.2,
            'payment_history_length': 5.0,
            'number_of_credit_cards': 2,
            'loan_amount': 15000.0,
            'loan_term': 36,
            'number_of_dependents': 1,
            'number_of_open_accounts': 3,
            'number_of_credit_inquiries': 1,
            'months_employed': 24,
            'employment_status': 'employed',
            'education_level': 'bachelor',
            'marital_status': 'single',
            'home_ownership': 'rent',
            'loan_purpose': 'debt_consolidation',
            'credit_history_length': 'medium'
        }
        
        # Devrait lever une exception
        with pytest.raises(Exception):
            CreditDataSchema(**invalid_data)

if __name__ == "__main__":
    pytest.main([__file__])



