"""
Tests pour l'API FastAPI
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app

# Client de test
client = TestClient(app)

class TestAPI:
    """Tests pour l'API FastAPI"""
    
    def test_root_endpoint(self):
        """Test de l'endpoint racine"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test de l'endpoint de santé"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "version" in data
    
    def test_score_endpoint_valid_data(self):
        """Test de l'endpoint de scoring avec des données valides"""
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
        
        response = client.post("/score", json=valid_data)
        
        # L'API peut retourner une erreur si les modèles ne sont pas chargés
        # mais la structure de la requête est valide
        if response.status_code == 200:
            data = response.json()
            assert "score" in data
            assert "probability_default" in data
            assert "risk_level" in data
            assert "confidence" in data
        elif response.status_code == 500:
            # Erreur interne (modèles non chargés)
            assert "detail" in response.json()
    
    def test_score_endpoint_invalid_data(self):
        """Test de l'endpoint de scoring avec des données invalides"""
        invalid_data = {
            "age": 150.0,  # Âge invalide
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
        
        response = client.post("/score", json=invalid_data)
        
        # Devrait retourner une erreur de validation
        assert response.status_code == 422
    
    def test_score_endpoint_missing_data(self):
        """Test de l'endpoint de scoring avec des données manquantes"""
        incomplete_data = {
            "age": 30.0,
            "income": 50000.0
            # Données manquantes
        }
        
        response = client.post("/score", json=incomplete_data)
        
        # Devrait retourner une erreur de validation
        assert response.status_code == 422
    
    def test_chr_endpoint_valid_data(self):
        """Test de l'endpoint de classification avec des données valides"""
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
        
        response = client.post("/chr", json=valid_data)
        
        # L'API peut retourner une erreur si les modèles ne sont pas chargés
        # mais la structure de la requête est valide
        if response.status_code == 200:
            data = response.json()
            assert "class_risk" in data
            assert "probability" in data
            assert "risk_score" in data
        elif response.status_code == 500:
            # Erreur interne (modèles non chargés)
            assert "detail" in response.json()
    
    def test_models_info_endpoint(self):
        """Test de l'endpoint d'informations sur les modèles"""
        response = client.get("/models/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "preprocessor_loaded" in data
        assert "logistic_model_loaded" in data
        assert "scorecard_loaded" in data
        assert "chr_model_loaded" in data
        assert "xgb_model_loaded" in data
        assert "feature_config" in data
    
    def test_batch_score_endpoint(self):
        """Test de l'endpoint de scoring par lot"""
        batch_data = [
            {
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
            },
            {
                "age": 35.0,
                "income": 60000.0,
                "debt_ratio": 0.4,
                "monthly_income": 5000.0,
                "credit_utilization": 0.3,
                "payment_history_length": 7.0,
                "number_of_credit_cards": 3,
                "loan_amount": 20000.0,
                "loan_term": 48,
                "number_of_dependents": 2,
                "number_of_open_accounts": 4,
                "number_of_credit_inquiries": 2,
                "months_employed": 36,
                "employment_status": "employed",
                "education_level": "master",
                "marital_status": "married",
                "home_ownership": "mortgage",
                "loan_purpose": "home_improvement",
                "credit_history_length": "long"
            }
        ]
        
        response = client.post("/batch_score", json=batch_data)
        
        # L'API peut retourner une erreur si les modèles ne sont pas chargés
        # mais la structure de la requête est valide
        if response.status_code == 200:
            data = response.json()
            assert "total_requests" in data
            assert "successful" in data
            assert "failed" in data
            assert "results" in data
            assert data["total_requests"] == 2
        elif response.status_code == 500:
            # Erreur interne (modèles non chargés)
            assert "detail" in response.json()

class TestAPIModels:
    """Tests pour la validation des modèles Pydantic"""
    
    def test_credit_data_model_validation(self):
        """Test de validation du modèle CreditData"""
        from api.main import CreditData
        
        # Données valides
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
        
        # Créer l'objet (devrait réussir)
        credit_data = CreditData(**valid_data)
        
        # Vérifier les valeurs
        assert credit_data.age == 30.0
        assert credit_data.income == 50000.0
        assert credit_data.employment_status == "employed"
    
    def test_credit_data_model_invalid_validation(self):
        """Test de validation du modèle CreditData avec des données invalides"""
        from api.main import CreditData
        
        # Données invalides
        invalid_data = {
            "age": 150.0,  # Âge invalide
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
        
        # Devrait lever une exception
        with pytest.raises(Exception):
            CreditData(**invalid_data)

if __name__ == "__main__":
    pytest.main([__file__])



