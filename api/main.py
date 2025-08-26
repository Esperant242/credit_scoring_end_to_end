"""
API FastAPI pour le scoring de crédit
Endpoints: /health, /score, /chr
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import sys
import logging

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_model_paths, get_feature_config
from src.utils.logging import setup_logger

# Configuration du logger
logger = setup_logger("api", log_file="api.log")

# Créer l'application FastAPI
app = FastAPI(
    title="Credit Scoring API",
    description="API pour le scoring de crédit et la classification en classes de risque",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic pour la validation des données
class CreditData(BaseModel):
    """Données de crédit pour le scoring"""
    age: float = Field(..., ge=18, le=100, description="Âge du demandeur")
    income: float = Field(..., ge=0, le=1000000, description="Revenu annuel")
    debt_ratio: float = Field(..., ge=0, le=10, description="Ratio dette/revenu")
    monthly_income: float = Field(..., ge=0, le=100000, description="Revenu mensuel")
    credit_utilization: float = Field(..., ge=0, le=1, description="Utilisation du crédit")
    payment_history_length: float = Field(..., ge=0, le=50, description="Longueur historique paiement")
    number_of_credit_cards: int = Field(..., ge=0, le=20, description="Nombre de cartes de crédit")
    loan_amount: float = Field(..., ge=1000, le=1000000, description="Montant du prêt")
    loan_term: int = Field(..., ge=12, le=360, description="Durée du prêt en mois")
    number_of_dependents: int = Field(..., ge=0, le=10, description="Nombre de personnes à charge")
    number_of_open_accounts: int = Field(..., ge=0, le=50, description="Nombre de comptes ouverts")
    number_of_credit_inquiries: int = Field(..., ge=0, le=20, description="Nombre de demandes de crédit")
    months_employed: int = Field(..., ge=0, le=600, description="Mois d'emploi")
    employment_status: str = Field(..., description="Statut d'emploi")
    education_level: str = Field(..., description="Niveau d'éducation")
    marital_status: str = Field(..., description="Statut marital")
    home_ownership: str = Field(..., description="Propriété du logement")
    loan_purpose: str = Field(..., description="Objectif du prêt")
    credit_history_length: str = Field(..., description="Longueur historique crédit")

class ScoringResponse(BaseModel):
    """Réponse du scoring"""
    score: int = Field(..., description="Score de crédit (0-1000)")
    probability_default: float = Field(..., description="Probabilité de défaut")
    risk_level: str = Field(..., description="Niveau de risque")
    confidence: float = Field(..., description="Niveau de confiance du score")

class CHRResponse(BaseModel):
    """Réponse de classification en classes de risque"""
    class_risk: str = Field(..., description="Classe de risque")
    probability: float = Field(..., description="Probabilité d'appartenance à la classe")
    risk_score: int = Field(..., description="Score de risque (1-5)")

class HealthResponse(BaseModel):
    """Réponse de santé de l'API"""
    status: str = Field(..., description="Statut de l'API")
    timestamp: str = Field(..., description="Horodatage de la vérification")
    models_loaded: bool = Field(..., description="Modèles chargés avec succès")
    version: str = Field(..., description="Version de l'API")

# Variables globales pour les modèles
preprocessor = None
logistic_model = None
scorecard_model = None
chr_model = None
xgb_model = None

def load_models():
    """Charge tous les modèles nécessaires"""
    global preprocessor, logistic_model, scorecard_model, chr_model, xgb_model
    
    try:
        model_paths = get_model_paths()
        
        # Charger le préprocesseur
        if model_paths['preprocessor'].exists():
            with open(model_paths['preprocesseur'], 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info("✅ Préprocesseur chargé")
        else:
            logger.warning("⚠️ Préprocesseur non trouvé")
        
        # Charger le modèle logistique
        if model_paths['logistic'].exists():
            with open(model_paths['logistic'], 'rb') as f:
                logistic_model = pickle.load(f)
            logger.info("✅ Modèle logistique chargé")
        else:
            logger.warning("⚠️ Modèle logistique non trouvé")
        
        # Charger le scorecard
        if model_paths['scorecard'].exists():
            with open(model_paths['scorecard'], 'r') as f:
                scorecard_model = json.load(f)
            logger.info("✅ Scorecard chargé")
        else:
            logger.warning("⚠️ Scorecard non trouvé")
        
        # Charger le modèle CHR
        if model_paths['chr'].exists():
            with open(model_paths['chr'], 'rb') as f:
                chr_model = pickle.load(f)
            logger.info("✅ Modèle CHR chargé")
        else:
            logger.warning("⚠️ Modèle CHR non trouvé")
        
        # Charger le modèle XGBoost
        if model_paths['xgb'].exists():
            with open(model_paths['xgb'], 'rb') as f:
                xgb_model = pickle.load(f)
            logger.info("✅ Modèle XGBoost chargé")
        else:
            logger.warning("⚠️ Modèle XGBoost non trouvé")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
        return False

def preprocess_data(data: CreditData) -> np.ndarray:
    """Préprocesse les données d'entrée"""
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Préprocesseur non chargé")
    
    # Convertir en DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Ajouter une colonne cible factice pour l'encodeur cible
    df['target'] = 0
    
    # Préprocesser
    try:
        X_transformed = preprocessor.transform(df)
        return X_transformed
    except Exception as e:
        logger.error(f"Erreur de preprocessing: {e}")
        raise HTTPException(status_code=400, detail=f"Erreur de preprocessing: {str(e)}")

def calculate_score(probability: float) -> int:
    """Calcule le score de 0 à 1000"""
    # Transformation logit vers score
    # Score = 1000 - (logit * 100)
    logit = np.log(probability / (1 - probability))
    score = int(1000 - (logit * 100))
    return max(0, min(1000, score))

def get_risk_level(score: int) -> str:
    """Détermine le niveau de risque basé sur le score"""
    if score >= 800:
        return "Excellent"
    elif score >= 700:
        return "Bon"
    elif score >= 600:
        return "Moyen"
    elif score >= 500:
        return "Faible"
    else:
        return "Très faible"

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage de l'API"""
    logger.info("🚀 Démarrage de l'API de scoring de crédit")
    load_models()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint racine"""
    return {
        "message": "API de Scoring de Crédit",
        "version": "1.0.0",
        "endpoints": "/health, /score, /chr"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état de l'API"""
    from datetime import datetime
    
    models_loaded = all([
        preprocessor is not None,
        logistic_model is not None,
        scorecard_model is not None
    ])
    
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        version="1.0.0"
    )

@app.post("/score", response_model=ScoringResponse)
async def calculate_credit_score(data: CreditData):
    """Calcule le score de crédit"""
    try:
        logger.info(f"Calcul du score pour: {data.age} ans, revenu: {data.income}")
        
        # Préprocesser les données
        X_transformed = preprocess_data(data)
        
        # Prédiction avec le modèle logistique
        if logistic_model is not None:
            probability_default = logistic_model.predict_proba(X_transformed)[0, 1]
        else:
            # Fallback: probabilité basée sur des règles métier
            probability_default = 0.1  # Valeur par défaut
        
        # Calculer le score
        score = calculate_score(probability_default)
        risk_level = get_risk_level(score)
        
        # Niveau de confiance (basé sur la qualité des données)
        confidence = 0.85  # À améliorer avec des métriques de qualité
        
        logger.info(f"Score calculé: {score}, Risque: {risk_level}")
        
        return ScoringResponse(
            score=score,
            probability_default=float(probability_default),
            risk_level=risk_level,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul du score: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/chr", response_model=CHRResponse)
async def classify_risk_class(data: CreditData):
    """Classifie en classes de risque"""
    try:
        logger.info(f"Classification en classe de risque pour: {data.age} ans")
        
        # Préprocesser les données
        X_transformed = preprocess_data(data)
        
        # Classification avec le modèle CHR
        if chr_model is not None:
            class_prediction = chr_model.predict(X_transformed)[0]
            class_probability = np.max(chr_model.predict_proba(X_transformed))
        else:
            # Fallback: classification basée sur des règles
            class_prediction = "Moyen"
            class_probability = 0.7
        
        # Mapping vers score de risque
        risk_mapping = {
            "Très faible": 1,
            "Faible": 2,
            "Moyen": 3,
            "Élevé": 4,
            "Très élevé": 5
        }
        risk_score = risk_mapping.get(class_prediction, 3)
        
        logger.info(f"Classe de risque: {class_prediction}, Score: {risk_score}")
        
        return CHRResponse(
            class_risk=class_prediction,
            probability=float(class_probability),
            risk_score=risk_score
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la classification: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/batch_score")
async def batch_score(data_list: List[CreditData]):
    """Calcul de scores pour un lot de données"""
    try:
        logger.info(f"Calcul de scores pour {len(data_list)} demandes")
        
        results = []
        for i, data in enumerate(data_list):
            try:
                score_response = await calculate_credit_score(data)
                results.append({
                    "index": i,
                    "status": "success",
                    "result": score_response.dict()
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "total_requests": len(data_list),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement par lot: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/models/info")
async def get_models_info():
    """Informations sur les modèles chargés"""
    return {
        "preprocessor_loaded": preprocessor is not None,
        "logistic_model_loaded": logistic_model is not None,
        "scorecard_loaded": scorecard_model is not None,
        "chr_model_loaded": chr_model is not None,
        "xgb_model_loaded": xgb_model is not None,
        "feature_config": get_feature_config()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



