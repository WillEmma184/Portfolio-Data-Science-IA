import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def evaluation_churn():
    print("--- Évaluation du modèle de Churn ---")
    
    # 1. Chargement des données de test (mêmes conditions que train.py)
    data_path = os.path.join(os.path.dirname(__file__), '../data/churn_bancaire_50k.csv')
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Chargement du modèle optimisé
    model_path = os.path.join(os.path.dirname(__file__), '../models/xgboost_churn_model.pkl')
    if not os.path.exists(model_path):
        print("Erreur : Le modèle entraîné est introuvable.")
        return
        
    model = joblib.load(model_path)
    
    # 3. Prédictions
    y_pred = model.predict(X_test)
    
    # 4. Génération des métriques
    rapport_txt = classification_report(y_test, y_pred)
    matrice = confusion_matrix(y_test, y_pred)
    
    # 5. Construction et sauvegarde du rapport métier
    metrics_summary = f"""==================================================
RAPPORT DE PERFORMANCE - PRÉDICTION CHURN CLIENT (XGBOOST + OPTUNA)
==================================================

Format du jeu de test : {X_test.shape[0]} clients évalues.

1. METRIQUES GLOBALES :
{rapport_txt}

2. MATRICE DE CONFUSION :
[[Vrais Restes (TN), Faux Partis (FP)]
 [Faux Restes (FN), Vrais Partis (TP)]]

{matrice}

==================================================
Conclusion : Le modèle est prêt pour la segmentation marketing ciblée.
"""
    
    output_path = os.path.join(os.path.dirname(__file__), '../models/rapport_performance.txt')
    with open(output_path, "w") as f:
        f.write(metrics_summary)
        
    print(f"Rapport de performance généré avec succès dans : {output_path}")

if __name__ == "__main__":
    evaluation_churn()