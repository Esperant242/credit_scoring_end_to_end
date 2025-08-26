"""
Application Streamlit pour le scoring de crédit
Interface utilisateur simple avec upload de fichiers et formulaire
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_feature_config

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🏦 Application de Scoring de Crédit")
st.markdown("---")

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Vérifie la santé de l'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except:
        return False, None

def calculate_score_api(data):
    """Calcule le score via l'API"""
    try:
        response = requests.post(f"{API_BASE_URL}/score", json=data, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.text
    except Exception as e:
        return False, str(e)

def classify_risk_api(data):
    """Classifie en classe de risque via l'API"""
    try:
        response = requests.post(f"{API_BASE_URL}/chr", json=data, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.text
    except Exception as e:
        return False, str(e)

# Sidebar
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("---")

# Vérification de l'API
api_healthy, api_status = check_api_health()
if api_healthy:
    st.sidebar.success("✅ API connectée")
    st.sidebar.json(api_status)
else:
    st.sidebar.error("❌ API non connectée")
    st.sidebar.info("Lancez l'API avec: `make api`")

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs(["📊 Scoring", "📁 Upload CSV", "📈 Visualisations", "ℹ️ À propos"])

with tab1:
    st.header("🎯 Calcul de Score de Crédit")
    
    # Formulaire de saisie
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations personnelles")
        age = st.slider("Âge", 18, 100, 30)
        employment_status = st.selectbox(
            "Statut d'emploi",
            ["employed", "unemployed", "self-employed", "retired", "student"]
        )
        education_level = st.selectbox(
            "Niveau d'éducation",
            ["high_school", "bachelor", "master", "phd", "other"]
        )
        marital_status = st.selectbox(
            "Statut marital",
            ["single", "married", "divorced", "widowed"]
        )
        home_ownership = st.selectbox(
            "Propriété du logement",
            ["own", "rent", "mortgage", "other"]
        )
    
    with col2:
        st.subheader("Informations financières")
        income = st.number_input("Revenu annuel ($)", 0, 1000000, 50000, step=1000)
        monthly_income = st.number_input("Revenu mensuel ($)", 0, 100000, 4000, step=100)
        debt_ratio = st.slider("Ratio dette/revenu", 0.0, 10.0, 0.3, step=0.1)
        credit_utilization = st.slider("Utilisation du crédit", 0.0, 1.0, 0.2, step=0.05)
        loan_amount = st.number_input("Montant du prêt ($)", 1000, 1000000, 15000, step=1000)
        loan_term = st.selectbox("Durée du prêt (mois)", [12, 24, 36, 48, 60])
    
    # Informations supplémentaires
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Historique de crédit")
        payment_history_length = st.slider("Longueur historique paiement", 0, 50, 5)
        number_of_credit_cards = st.slider("Nombre de cartes de crédit", 0, 20, 2)
        number_of_open_accounts = st.slider("Nombre de comptes ouverts", 0, 50, 3)
        credit_history_length = st.selectbox(
            "Longueur historique crédit",
            ["new", "short", "medium", "long", "excellent"]
        )
    
    with col4:
        st.subheader("Autres informations")
        number_of_dependents = st.slider("Nombre de personnes à charge", 0, 10, 1)
        number_of_credit_inquiries = st.slider("Nombre de demandes de crédit", 0, 20, 1)
        months_employed = st.slider("Mois d'emploi", 0, 600, 24)
        loan_purpose = st.selectbox(
            "Objectif du prêt",
            ["debt_consolidation", "home_improvement", "major_purchase", "medical", "other"]
        )
    
    # Bouton de calcul
    if st.button("🚀 Calculer le Score", type="primary"):
        if not api_healthy:
            st.error("❌ API non disponible. Vérifiez la connexion.")
        else:
            with st.spinner("Calcul en cours..."):
                # Préparer les données
                data = {
                    "age": float(age),
                    "income": float(income),
                    "debt_ratio": float(debt_ratio),
                    "monthly_income": float(monthly_income),
                    "credit_utilization": float(credit_utilization),
                    "payment_history_length": float(payment_history_length),
                    "number_of_credit_cards": int(number_of_credit_cards),
                    "loan_amount": float(loan_amount),
                    "loan_term": int(loan_term),
                    "number_of_dependents": int(number_of_dependents),
                    "number_of_open_accounts": int(number_of_open_accounts),
                    "number_of_credit_inquiries": int(number_of_credit_inquiries),
                    "months_employed": int(months_employed),
                    "employment_status": employment_status,
                    "education_level": education_level,
                    "marital_status": marital_status,
                    "home_ownership": home_ownership,
                    "loan_purpose": loan_purpose,
                    "credit_history_length": credit_history_length
                }
                
                # Calculer le score
                success, result = calculate_score_api(data)
                
                if success:
                    st.success("✅ Score calculé avec succès!")
                    
                    # Affichage des résultats
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.metric("Score de crédit", f"{result['score']}/1000")
                        st.metric("Niveau de risque", result['risk_level'])
                    
                    with col_result2:
                        st.metric("Probabilité de défaut", f"{result['probability_default']:.1%}")
                        st.metric("Confiance", f"{result['confidence']:.1%}")
                    
                    # Barre de score
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Score de Crédit"},
                        delta={'reference': 500},
                        gauge={
                            'axis': {'range': [None, 1000]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 500], 'color': "lightgray"},
                                {'range': [500, 600], 'color': "yellow"},
                                {'range': [600, 700], 'color': "orange"},
                                {'range': [700, 800], 'color': "lightgreen"},
                                {'range': [800, 1000], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 500
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification en classe de risque
                    st.subheader("🎯 Classification en Classe de Risque")
                    success_chr, result_chr = classify_risk_api(data)
                    
                    if success_chr:
                        col_chr1, col_chr2, col_chr3 = st.columns(3)
                        
                        with col_chr1:
                            st.metric("Classe de risque", result_chr['class_risk'])
                        
                        with col_chr2:
                            st.metric("Score de risque", f"{result_chr['risk_score']}/5")
                        
                        with col_chr3:
                            st.metric("Confiance", f"{result_chr['probability']:.1%}")
                        
                        # Gauge de risque
                        risk_colors = ["green", "lightgreen", "yellow", "orange", "red"]
                        fig_risk = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result_chr['risk_score'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Score de Risque"},
                            gauge={
                                'axis': {'range': [None, 5]},
                                'bar': {'color': risk_colors[result_chr['risk_score']-1]},
                                'steps': [
                                    {'range': [0, 1], 'color': "lightgray"},
                                    {'range': [1, 2], 'color': "lightgreen"},
                                    {'range': [2, 3], 'color': "yellow"},
                                    {'range': [3, 4], 'color': "orange"},
                                    {'range': [4, 5], 'color': "red"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig_risk, use_container_width=True)
                    else:
                        st.error(f"❌ Erreur lors de la classification: {result_chr}")
                else:
                    st.error(f"❌ Erreur lors du calcul: {result}")

with tab2:
    st.header("📁 Upload de Fichier CSV")
    
    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV",
        type=['csv'],
        help="Le fichier doit contenir les colonnes requises pour le scoring"
    )
    
    if uploaded_file is not None:
        try:
            # Lire le fichier
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Fichier chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Aperçu des données
            st.subheader("🔍 Aperçu des données")
            st.dataframe(df.head())
            
            # Vérification des colonnes
            st.subheader("📋 Vérification des colonnes")
            required_columns = [
                'age', 'income', 'debt_ratio', 'monthly_income', 'credit_utilization',
                'payment_history_length', 'number_of_credit_cards', 'loan_amount', 'loan_term',
                'number_of_dependents', 'number_of_open_accounts', 'number_of_credit_inquiries',
                'months_employed', 'employment_status', 'education_level', 'marital_status',
                'home_ownership', 'loan_purpose', 'credit_history_length'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Colonnes manquantes: {missing_columns}")
            else:
                st.success("✅ Toutes les colonnes requises sont présentes")
                
                # Bouton de traitement par lot
                if st.button("🚀 Traiter le fichier complet", type="primary"):
                    if not api_healthy:
                        st.error("❌ API non disponible")
                    else:
                        with st.spinner("Traitement en cours..."):
                            # Traiter chaque ligne
                            results = []
                            progress_bar = st.progress(0)
                            
                            for idx, row in df.iterrows():
                                try:
                                    # Convertir en dictionnaire
                                    data = row.to_dict()
                                    
                                    # Calculer le score
                                    success, result = calculate_score_api(data)
                                    
                                    if success:
                                        results.append({
                                            'index': idx,
                                            'score': result['score'],
                                            'risk_level': result['risk_level'],
                                            'probability_default': result['probability_default']
                                        })
                                    else:
                                        results.append({
                                            'index': idx,
                                            'error': result
                                        })
                                    
                                    # Mettre à jour la barre de progression
                                    progress_bar.progress((idx + 1) / len(df))
                                    
                                except Exception as e:
                                    results.append({
                                        'index': idx,
                                        'error': str(e)
                                    })
                    
                            # Afficher les résultats
                            st.success(f"✅ Traitement terminé: {len(results)} lignes traitées")
                            
                            # Créer un DataFrame des résultats
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df)
                            
                            # Statistiques
                            if 'score' in results_df.columns:
                                st.subheader("📊 Statistiques des scores")
                                col_stats1, col_stats2, col_stats3 = st.columns(3)
                                
                                with col_stats1:
                                    st.metric("Score moyen", f"{results_df['score'].mean():.0f}")
                                    st.metric("Score médian", f"{results_df['score'].median():.0f}")
                                
                                with col_stats2:
                                    st.metric("Score min", f"{results_df['score'].min():.0f}")
                                    st.metric("Score max", f"{results_df['score'].max():.0f}")
                                
                                with col_stats3:
                                    st.metric("Écart-type", f"{results_df['score'].std():.0f}")
                                
                                # Distribution des scores
                                fig_dist = px.histogram(
                                    results_df, 
                                    x='score', 
                                    nbins=20,
                                    title="Distribution des scores de crédit"
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)
                                
                                # Distribution des niveaux de risque
                                risk_counts = results_df['risk_level'].value_counts()
                                fig_risk_dist = px.pie(
                                    values=risk_counts.values,
                                    names=risk_counts.index,
                                    title="Répartition des niveaux de risque"
                                )
                                st.plotly_chart(fig_risk_dist, use_container_width=True)
                            
                            # Télécharger les résultats
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger les résultats",
                                data=csv,
                                file_name=f"credit_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier: {e}")

with tab3:
    st.header("📈 Visualisations et Analyses")
    
    if not api_healthy:
        st.warning("⚠️ Connectez-vous d'abord à l'API pour voir les visualisations")
    else:
        st.info("📊 Les visualisations seront affichées ici après le calcul de scores")
        
        # Exemple de visualisations
        st.subheader("🎯 Exemples de visualisations")
        
        # Gauge de score
        fig_example = go.Figure(go.Indicator(
            mode="gauge+number",
            value=750,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Score de Crédit (Exemple)"},
            gauge={
                'axis': {'range': [None, 1000]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 500], 'color': "lightgray"},
                    {'range': [500, 600], 'color': "yellow"},
                    {'range': [600, 700], 'color': "orange"},
                    {'range': [700, 800], 'color': "lightgreen"},
                    {'range': [800, 1000], 'color': "green"}
                ]
            }
        ))
        st.plotly_chart(fig_example, use_container_width=True)

with tab4:
    st.header("ℹ️ À propos de l'application")
    
    st.markdown("""
    ## 🏦 Application de Scoring de Crédit
    
    Cette application permet de calculer des scores de crédit et de classifier les demandeurs en classes de risque.
    
    ### 🚀 Fonctionnalités
    
    - **Calcul de score** : Score de 0 à 1000 basé sur les caractéristiques du demandeur
    - **Classification de risque** : 5 classes de risque (Très faible à Très élevé)
    - **Upload de fichiers** : Traitement par lot de fichiers CSV
    - **Visualisations** : Graphiques et analyses des résultats
    
    ### 🔧 Technologies utilisées
    
    - **Frontend** : Streamlit
    - **Backend** : FastAPI
    - **Machine Learning** : Scikit-learn, XGBoost
    - **Visualisation** : Plotly
    
    ### 📊 Métriques de scoring
    
    - **Score 800-1000** : Excellent (risque très faible)
    - **Score 700-799** : Bon (risque faible)
    - **Score 600-699** : Moyen (risque modéré)
    - **Score 500-599** : Faible (risque élevé)
    - **Score 0-499** : Très faible (risque très élevé)
    
    ### 🎯 Utilisation
    
    1. Remplissez le formulaire dans l'onglet "Scoring"
    2. Ou uploadez un fichier CSV dans l'onglet "Upload CSV"
    3. Consultez les résultats et visualisations
    4. Téléchargez les résultats si nécessaire
    
    ### 📞 Support
    
    Pour toute question ou problème, contactez l'équipe de développement.
    """)

# Footer
st.markdown("---")
st.markdown(
    "🏦 **Credit Scoring App** - Développé avec ❤️ par l'équipe Data Science"
)


