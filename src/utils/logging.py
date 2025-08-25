"""
Configuration du logging pour le projet de scoring de crédit
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = "credit_scoring",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure et retourne un logger personnalisé
    
    Args:
        name: Nom du logger
        level: Niveau de logging
        log_file: Chemin vers le fichier de log (optionnel)
        console_output: Si True, affiche les logs dans la console
    
    Returns:
        Logger configuré
    """
    # Créer le logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Éviter la duplication des handlers
    if logger.handlers:
        return logger
    
    # Format du log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler pour la console
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Handler pour le fichier
    if log_file:
        # Créer le dossier de logs s'il n'existe pas
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "credit_scoring") -> logging.Logger:
    """
    Retourne un logger existant ou en crée un nouveau
    
    Args:
        name: Nom du logger
        
    Returns:
        Logger
    """
    return logging.getLogger(name)

# Logger par défaut
default_logger = setup_logger()

def log_function_call(func):
    """
    Décorateur pour logger les appels de fonction
    
    Args:
        func: Fonction à décorer
        
    Returns:
        Fonction décorée
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.info(f"Appel de {func.__name__} avec args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Fonction {func.__name__} terminée avec succès")
            return result
        except Exception as e:
            logger.error(f"Erreur dans {func.__name__}: {str(e)}")
            raise
    return wrapper

def log_execution_time(func):
    """
    Décorateur pour logger le temps d'exécution des fonctions
    
    Args:
        func: Fonction à décorer
        
    Returns:
        Fonction décorée
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = time.time()
        logger.info(f"Début de l'exécution de {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Fin de l'exécution de {func.__name__} en {execution_time:.2f} secondes")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Erreur dans {func.__name__} après {execution_time:.2f} secondes: {str(e)}")
            raise
    
    return wrapper

# Exemple d'utilisation
if __name__ == "__main__":
    logger = setup_logger("test", logging.DEBUG)
    logger.debug("Message de debug")
    logger.info("Message d'information")
    logger.warning("Message d'avertissement")
    logger.error("Message d'erreur")
    
    # Test du décorateur
    @log_function_call
    def test_function(x, y):
        return x + y
    
    result = test_function(5, 3)
    print(f"Résultat: {result}")



