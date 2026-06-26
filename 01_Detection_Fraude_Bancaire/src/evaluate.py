import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def generate_report():
    print("--- Génération du Rapport Métier ---")
    
    # 1. Chargement des données et du modèle
    data_path = os.path.join(os.path.dirname(__file__), '../data/creditcard.csv')
    model_path = os.path.join(os.path.dirname(__file__), '../models/xgboost_fraud_model.pkl')
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    # 2. Calcul des métriques
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. Écriture du rapport d'aide à la décision
    report_txt_path = os.path.join(os.path.dirname(__file__), '../models/rapport_performance.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("RAPPORT MÉTIER : DÉTECTION DE FRAUDE BANCAIRE\n")
        f.write("==================================================\n\n")
        f.write(f"Modèle évalué : XGBoost + SMOTE\n")
        f.write(f"Nombre de transactions testées : {len(y_test)}\n\n")
        f.write("--- MÈTRIQUES CLÉS ---\n")
        f.write(f"Rappel (Recall - Capacité à détecter la fraude) : {report_dict['1']['recall']:.2%}\n")
        f.write(f"Précision (Fiabilité des alertes de fraude) : {report_dict['1']['precision']:.2%}\n")
        f.write(f"Score F1 (Équilibre) : {report_dict['1']['f1-score']:.2%}\n\n")
        f.write("--- MATRICE DE CONFUSION ---\n")
        f.write(f"Vrais Légitimes : {cm[0][0]}  | Faux Positifs (Fausses alertes) : {cm[0][1]}\n")
        f.write(f"Faux Négatifs (Fraudes ratées) : {cm[1][0]} | Vrais Positifs (Fraudes détectées) : {cm[1][1]}\n\n")
        f.write("--- CONCLUSION AIDE À LA DÉCISION ---\n")
        f.write("Le modèle priorise la couverture du risque (Rappel élevé).\n")
        f.write("Les variables V14, V4 et V12 impactent le plus les décisions selon l'analyse SHAP.\n")
        
    print(f"Rapport généré avec succès dans : {report_txt_path}")

if __name__ == "__main__":
    generate_report()