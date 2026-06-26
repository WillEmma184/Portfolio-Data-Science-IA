import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def run_pipeline():
    print("--- Début du pipeline d'entraînement ---")
    
    # 1. Chargement des données
    data_path = os.path.join(os.path.dirname(__file__), '../data/creditcard.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier {data_path} est introuvable.")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Équilibrage SMOTE (uniquement sur le train)
    print("Application de SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 4. Entraînement du modèle XGBoost
    print("Entraînement du modèle XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_res, y_train_res)
    
    # 5. Sauvegarde du modèle entraîné
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'xgboost_fraud_model.pkl')
    
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé avec succès dans : {model_path}")
    print("--- Fin du pipeline ---")

if __name__ == "__main__":
    run_pipeline()