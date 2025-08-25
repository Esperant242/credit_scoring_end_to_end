"""
Schéma et validation des données pour le scoring de crédit
"""
from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field, validator
import numpy as np

class CreditDataSchema(BaseModel):
    """Schéma de validation pour les données de crédit"""
    
    # Features numériques
    age: float = Field(..., ge=18, le=100, description="Âge du demandeur")
    income: float = Field(..., ge=0, le=1000000, description="Revenu annuel")
    debt_ratio: float = Field(..., ge=0, le=10, description="Ratio dette/revenu")
    monthly_income: float = Field(..., ge=0, le=100000, description="Revenu mensuel")
    credit_utilization: float = Field(..., ge=0, le=1, description="Utilisation du crédit (0-1)")
    payment_history_length: float = Field(..., ge=0, le=50, description="Longueur de l'historique de paiement")
    number_of_credit_cards: int = Field(..., ge=0, le=20, description="Nombre de cartes de crédit")
    loan_amount: float = Field(..., ge=1000, le=1000000, description="Montant du prêt demandé")
    loan_term: int = Field(..., ge=12, le=360, description="Durée du prêt en mois")
    
    # Features discrètes
    number_of_dependents: int = Field(..., ge=0, le=10, description="Nombre de personnes à charge")
    number_of_open_accounts: int = Field(..., ge=0, le=50, description="Nombre de comptes ouverts")
    number_of_credit_inquiries: int = Field(..., ge=0, le=20, description="Nombre de demandes de crédit")
    months_employed: int = Field(..., ge=0, le=600, description="Mois d'emploi")
    
    # Features catégorielles
    employment_status: str = Field(..., description="Statut d'emploi")
    education_level: str = Field(..., description="Niveau d'éducation")
    marital_status: str = Field(..., description="Statut marital")
    home_ownership: str = Field(..., description="Propriété du logement")
    loan_purpose: str = Field(..., description="Objectif du prêt")
    credit_history_length: str = Field(..., description="Longueur de l'historique de crédit")
    
    @validator('employment_status')
    def validate_employment_status(cls, v):
        valid_values = ['employed', 'unemployed', 'self-employed', 'retired', 'student']
        if v not in valid_values:
            raise ValueError(f'employment_status doit être l\'un de: {valid_values}')
        return v
    
    @validator('education_level')
    def validate_education_level(cls, v):
        valid_values = ['high_school', 'bachelor', 'master', 'phd', 'other']
        if v not in valid_values:
            raise ValueError(f'education_level doit être l\'un de: {valid_values}')
        return v
    
    @validator('marital_status')
    def validate_marital_status(cls, v):
        valid_values = ['single', 'married', 'divorced', 'widowed']
        if v not in valid_values:
            raise ValueError(f'marital_status doit être l\'un de: {valid_values}')
        return v
    
    @validator('home_ownership')
    def validate_home_ownership(cls, v):
        valid_values = ['own', 'rent', 'mortgage', 'other']
        if v not in valid_values:
            raise ValueError(f'home_ownership doit être l\'un de: {valid_values}')
        return v
    
    @validator('loan_purpose')
    def validate_loan_purpose(cls, v):
        valid_values = ['debt_consolidation', 'home_improvement', 'major_purchase', 'medical', 'other']
        if v not in valid_values:
            raise ValueError(f'loan_purpose doit être l\'un de: {valid_values}')
        return v
    
    @validator('credit_history_length')
    def validate_credit_history_length(cls, v):
        valid_values = ['new', 'short', 'medium', 'long', 'excellent']
        if v not in valid_values:
            raise ValueError(f'credit_history_length doit être l\'un de: {valid_values}')
        return v

class CreditDataValidator:
    """Classe pour valider les données de crédit"""
    
    def __init__(self):
        self.schema = CreditDataSchema
        self.errors = []
    
    def validate_row(self, row: pd.Series) -> bool:
        """
        Valide une ligne de données
        
        Args:
            row: Ligne de données pandas
            
        Returns:
            True si valide, False sinon
        """
        try:
            # Convertir en dictionnaire
            data_dict = row.to_dict()
            
            # Valider avec pydantic
            validated_data = self.schema(**data_dict)
            return True
        except Exception as e:
            self.errors.append(f"Ligne {row.name}: {str(e)}")
            return False
    
    def validate_dataframe(self, df: pd.DataFrame) -> tuple[bool, List[str]]:
        """
        Valide un DataFrame complet
        
        Args:
            df: DataFrame à valider
            
        Returns:
            Tuple (is_valid, errors)
        """
        self.errors = []
        valid_rows = 0
        
        for idx, row in df.iterrows():
            if self.validate_row(row):
                valid_rows += 1
        
        is_valid = valid_rows == len(df)
        return is_valid, self.errors
    
    def get_validation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Retourne un résumé de la validation
        
        Args:
            df: DataFrame validé
            
        Returns:
            Résumé de la validation
        """
        is_valid, errors = self.validate_dataframe(df)
        
        return {
            "is_valid": is_valid,
            "total_rows": len(df),
            "valid_rows": len(df) - len(errors),
            "invalid_rows": len(errors),
            "error_rate": len(errors) / len(df) if len(df) > 0 else 0,
            "errors": errors[:10]  # Limiter à 10 erreurs pour l'affichage
        }

def get_expected_columns() -> Dict[str, List[str]]:
    """Retourne les colonnes attendues par type"""
    return {
        "numerical": [
            "age", "income", "debt_ratio", "monthly_income", "credit_utilization",
            "payment_history_length", "number_of_credit_cards", "loan_amount", "loan_term"
        ],
        "discrete": [
            "number_of_dependents", "number_of_open_accounts", 
            "number_of_credit_inquiries", "months_employed"
        ],
        "categorical": [
            "employment_status", "education_level", "marital_status",
            "home_ownership", "loan_purpose", "credit_history_length"
        ]
    }

def check_missing_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Vérifie les colonnes manquantes
    
    Args:
        df: DataFrame à vérifier
        
    Returns:
        Dictionnaire des colonnes manquantes par type
    """
    expected = get_expected_columns()
    all_expected = []
    for cols in expected.values():
        all_expected.extend(cols)
    
    missing = [col for col in all_expected if col not in df.columns]
    extra = [col for col in df.columns if col not in all_expected]
    
    return {
        "missing": missing,
        "extra": extra,
        "expected_count": len(all_expected),
        "actual_count": len(df.columns)
    }

def get_data_types_info() -> Dict[str, str]:
    """Retourne les types de données attendus"""
    return {
        "age": "float64",
        "income": "float64",
        "debt_ratio": "float64",
        "monthly_income": "float64",
        "credit_utilization": "float64",
        "payment_history_length": "float64",
        "number_of_credit_cards": "int64",
        "loan_amount": "float64",
        "loan_term": "int64",
        "number_of_dependents": "int64",
        "number_of_open_accounts": "int64",
        "number_of_credit_inquiries": "int64",
        "months_employed": "int64",
        "employment_status": "object",
        "education_level": "object",
        "marital_status": "object",
        "home_ownership": "object",
        "loan_purpose": "object",
        "credit_history_length": "object"
    }

if __name__ == "__main__":
    # Test du schéma
    print("Test du schéma de validation...")
    
    # Données de test valides
    valid_data = {
        "age": 30.0,
        "income": 50000.0,
        "debt_ratio": 0.3,
        "monthly_income": 4166.67,
        "credit_utilization": 0.2,
        "payment_history_length": 5.0,
        "number_of_credit_cards": 2,
        "loan_amount": 15000.0,
        "loan_term": 36,
        "number_of_dependents": 1,
        "number_of_open_accounts": 3,
        "number_of_credit_inquiries": 1,
        "months_employed": 24,
        "employment_status": "employed",
        "education_level": "bachelor",
        "marital_status": "single",
        "home_ownership": "rent",
        "loan_purpose": "debt_consolidation",
        "credit_history_length": "medium"
    }
    
    try:
        validated = CreditDataSchema(**valid_data)
        print("✅ Données valides!")
    except Exception as e:
        print(f"❌ Erreur de validation: {e}")
    
    # Test avec des données invalides
    invalid_data = valid_data.copy()
    invalid_data["age"] = 150  # Âge invalide
    
    try:
        validated = CreditDataSchema(**invalid_data)
        print("✅ Données valides!")
    except Exception as e:
        print(f"❌ Erreur de validation attendue: {e}")



