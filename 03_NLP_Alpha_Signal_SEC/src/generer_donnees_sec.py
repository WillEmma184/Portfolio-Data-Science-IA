import os
import pandas as pd
import numpy as np

def generer_filings_sec():
    print("--- Génération autonome du Dataset SEC Filings (5 000 rapports) ---")
    
    np.random.seed(42)
    n_samples = 5000
    
    # Lexiques de simulation financière
    phrases_pos = [
        "We achieved record revenue growth this quarter driven by strong demand.",
        "Strategic investments in AI infrastructure are yielding significant margin expansions.",
        "Management is highly optimistic about future cash flows and market expansion.",
        "Our balance sheet remains robust with solid liquidity and capital efficiency."
    ]
    
    phrases_neg = [
        "We face substantial macroeconomic headwinds and supply chain disruptions.",
        "Net income decreased due to unexpected regulatory compliance costs and inflation.",
        "Management notes potential liquidity risks outlined in our risk factors disclosure.",
        "Operating margins contracted significantly this fiscal year amid intense competition."
    ]
    
    phrases_neu = [
        "The company filed its standard quarterly report in compliance with SEC regulations.",
        "The consolidated financial statements are prepared in accordance with GAAP.",
        "There were no material changes to our critical accounting policies this period.",
        "We continue to evaluate our ongoing operations and business strategy as disclosed."
    ]
    
    # Génération aléatoire équilibrée
    data = []
    categories = ['Positive', 'Negative', 'Neutral']
    
    for i in range(n_samples):
        cat = np.random.choice(categories, p=[0.35, 0.35, 0.30])
        
        if cat == 'Positive':
            text = np.random.choice(phrases_pos)
        elif cat == 'Negative':
            text = np.random.choice(phrases_neg)
        else:
            text = np.random.choice(phrases_neu)
            
        # Simuler une colonne secteur d'activité (Tech, Finance, Energy, Healthcare)
        sector = np.random.choice(['Technology', 'Financials', 'Energy', 'Healthcare'])
        
        data.append({
            'Filing_ID': f"SEC-2025-{i:04d}",
            'Sector': sector,
            'Text_Filing': text,
            'Sentiment_Target': cat
        })
        
    df = pd.DataFrame(data)
    
    # Sauvegarde
    output_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sec_filings_5k.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Succès ! Dataset créé de façon autonome : {df.shape[0]} lignes générées.")
    print(f"Sauvegardé dans : {output_path}")

if __name__ == "__main__":
    generer_filings_sec()