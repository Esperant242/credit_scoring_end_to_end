#!/usr/bin/env python3
"""
Script de démarrage rapide pour le projet de scoring de crédit
"""
import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    """Exécute une commande avec affichage"""
    print(f"\n🚀 {description}...")
    print(f"Commande: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} terminé avec succès")
        if result.stdout:
            print("Sortie:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de {description}: {e}")
        if e.stderr:
            print("Erreur:", e.stderr)
        return False

def check_dependencies():
    """Vérifie les dépendances"""
    print("🔍 Vérification des dépendances...")
    
    try:
        import pandas
        import numpy
        import sklearn
        import fastapi
        import streamlit
        print("✅ Toutes les dépendances principales sont installées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("Installez les dépendances avec: pip install -r requirements.txt")
        return False

def main():
    """Fonction principale"""
    print("🏦 PROJET DE SCORING DE CRÉDIT")
    print("=" * 50)
    
    # Vérifier les dépendances
    if not check_dependencies():
        print("\n❌ Veuillez installer les dépendances avant de continuer")
        sys.exit(1)
    
    print("\n📋 Options disponibles:")
    print("1. Préparer les données")
    print("2. Créer les features")
    print("3. Lancer l'API")
    print("4. Lancer l'application Streamlit")
    print("5. Exécuter les tests")
    print("6. Pipeline complet")
    print("0. Quitter")
    
    while True:
        try:
            choice = input("\n🎯 Votre choix (0-6): ").strip()
            
            if choice == "0":
                print("👋 Au revoir!")
                break
            elif choice == "1":
                run_command("python src/data/prepare.py", "Préparation des données")
            elif choice == "2":
                run_command("python src/features/pipeline.py", "Création des features")
            elif choice == "3":
                print("\n🌐 Lancement de l'API...")
                print("L'API sera accessible sur: http://localhost:8000")
                print("Documentation: http://localhost:8000/docs")
                print("Appuyez sur Ctrl+C pour arrêter l'API")
                run_command("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000", "Lancement de l'API")
            elif choice == "4":
                print("\n📱 Lancement de l'application Streamlit...")
                print("L'application sera accessible sur: http://localhost:8501")
                print("Appuyez sur Ctrl+C pour arrêter l'application")
                run_command("streamlit run app/Home.py", "Lancement de Streamlit")
            elif choice == "5":
                run_command("pytest tests/ -v", "Exécution des tests")
            elif choice == "6":
                print("\n🔄 Pipeline complet...")
                commands = [
                    ("python src/data/prepare.py", "Préparation des données"),
                    ("python src/features/pipeline.py", "Création des features"),
                    ("pytest tests/ -v", "Exécution des tests")
                ]
                
                for command, description in commands:
                    if not run_command(command, description):
                        print(f"❌ Pipeline interrompu à l'étape: {description}")
                        break
                else:
                    print("✅ Pipeline complet terminé avec succès!")
            else:
                print("❌ Choix invalide. Veuillez entrer un nombre entre 0 et 6.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Opération interrompue par l'utilisateur")
            break
        except Exception as e:
            print(f"\n❌ Erreur inattendue: {e}")
            break

if __name__ == "__main__":
    main()



