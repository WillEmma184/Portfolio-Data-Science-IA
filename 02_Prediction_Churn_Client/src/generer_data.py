import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generation_dataset_churn():
    print("--- Génération autonome du Dataset Churn (50K clients) ---")
    
    # Génération de données de classification réalistes
    X, y = make_classification(
        n_samples=50000, 
        n_features=10, 
        n_informative=7, 
        n_redundant=3, 
        random_state=42,
        weights=[0.8, 0.2] # 20% de Churn (réaliste pour une banque)
    )
    
    # Transformation en DataFrame avec des noms de colonnes explicites
    columns = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Code', 'Gender_Code'
    ]
    
    df = pd.DataFrame(X, columns=columns)
    
    # Redimensionnement des variables pour coller à la réalité bancaire
    df['CreditScore'] = ((df['CreditScore'] - df['CreditScore'].min()) / (df['CreditScore'].max() - df['CreditScore'].min()) * (850 - 350) + 350).astype(int)
    df['Age'] = ((df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min()) * (80 - 18) + 18).astype(int)
    df['Tenure'] = ((df['Tenure'] - df['Tenure'].min()) / (df['Tenure'].max() - df['Tenure'].min()) * 10).astype(int)
    df['Balance'] = ((df['Balance'] - df['Balance'].min()) / (df['Balance'].max() - df['Balance'].min()) * 200000).round(2)
    df['NumOfProducts'] = np.random.choice([1, 2, 3, 4], size=50000, p=[0.5, 0.45, 0.04, 0.01])
    df['HasCrCard'] = np.random.choice([0, 1], size=50000, p=[0.3, 0.7])
    df['IsActiveMember'] = np.random.choice([0, 1], size=50000, p=[0.5, 0.5])
    df['EstimatedSalary'] = ((df['EstimatedSalary'] - df['EstimatedSalary'].min()) / (df['EstimatedSalary'].max() - df['EstimatedSalary'].min()) * 150000 + 15000).round(2)
    df['Geography_Code'] = np.random.choice([0, 1, 2], size=50000, p=[0.5, 0.25, 0.25]) # Ex: France, Espagne, Allemagne
    df['Gender_Code'] = np.random.choice([0, 1], size=50000)
    
    # Ajout de la cible Exited (Churn)
    df['Exited'] = y
    
    # Sauvegarde dans le dossier data/
    output_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'churn_bancaire_50k.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Succès ! Dataset créé de façon autonome : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
    print(f"Fichier sauvegardé dans : {output_path}")

if __name__ == "__main__":
    generation_dataset_churn()