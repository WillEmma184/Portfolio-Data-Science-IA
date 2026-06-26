import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
import joblib

# Désactiver les messages trop verbeux d'Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def entrainement_churn():
    print("--- Début de l'optimisation et de l'entraînement (XGBoost + Optuna) ---")
    
    # 1. Chargement des données
    data_path = os.path.join(os.path.dirname(__file__), '../data/churn_bancaire_50k.csv')
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Définition de l'objectif de recherche pour Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # On évalue sur le score global de précision (Accuracy) pour l'optimisation
        return model.score(X_test, y_test)

    # 3. Lancement de l'optimisation Optuna (5 essais rapides pour l'exemple)
    print("Recherche des meilleurs hyperparamètres avec Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    
    print(f"Meilleurs hyperparamètres trouvés : {study.best_params}")
    print(f"Meilleure précision ciblée : {study.best_value:.4f}")
    
    # 4. Entraînement du modèle final avec le meilleur combo
    print("\nEntraînement du modèle final...")
    best_model = xgb.XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss')
    best_model.fit(X_train, y_train)
    
    # 5. Sauvegarde du modèle
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'xgboost_churn_model.pkl')
    
    joblib.dump(best_model, model_path)
    print(f"Modèle XGBoost optimisé sauvegardé avec succès dans : {model_path}")

if __name__ == "__main__":
    entrainement_churn()