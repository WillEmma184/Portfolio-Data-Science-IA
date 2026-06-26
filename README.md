# Portfolio de Projets Data Science & Intelligence Artificielle

Ce dépôt rassemble les projets d'ingénierie et de recherche menés dans le cadre de mon cursus en Data Science et IA à l'EFREI Paris. L'objectif de ce portfolio est de démontrer une maîtrise complète du cycle de vie de la donnée : depuis le feature engineering avancé et la gestion de classes fortement déséquilibrées, jusqu'au déploiement d'architectures hybrides et à la validation statistique rigoureuse.

---

## Profil et Objectifs Professionnels

* **Position ciblée** : Stage de Data Scientist (Disponibilité : Août 2026)
* **Formation** : Cycle Ingénieur en Data Science & Intelligence Artificielle — EFREI Paris (4ème année)
* **Axes d'expertise** : Machine Learning tabulaire, Traitement du Langage Naturel (NLP), Architectures de Retrieval-Augmented Generation (RAG), MLOps et Validation Statistique.

---

## Compétences Techniques Répertoriées

* **Modélisation & Apprentissage** : Scikit-Learn, XGBoost, Random Forest, LightGBM, Rééchantillonnage (SMOTE), Clustering non-supervisé.
* **IA Générative & NLP** : LangChain, LlamaIndex, Encodage vectoriel (FAISS), Transformers (BERT, FinBERT), Re-ranking Cross-Encoder, Analyse de sentiments.
* **Frameworks & Industrialisation** : Python (Avancé), SQL (CTE, Window Functions), Optuna, Joblib, Docker.
* **Restitution & Dashboarding** : Streamlit, Dash, Plotly, Power BI.

---

## Vue d'Ensemble des Projets

### 01. Détection d'Anomalies et Fraude Bancaire
* **Problématique** : Identifier des transactions frauduleuses au sein d'un volume hautement déséquilibré de 284K lignes.
* **Approche** : Traitement du déséquilibre de classes par la méthode de suréchantillonnage SMOTE et optimisation d'un classifieur XGBoost. Intégration de l'explicabilité globale et locale du modèle via l'approche SHAP.
* **Résultat** : Atteinte d'une metric de performance de 0,98 pour l'Aire Sous la Courbe (AUC).

### 02. Prédiction du Churn Client
* **Problématique** : Anticiper le taux de défection (churn) pour un portefeuille bancaire de 50K clients.
* **Approche** : Conception d'un pipeline complet de Feature Engineering et clustering comportemental. Utilisation du framework Optuna pour l'optimisation automatisée et bayésienne des hyperparamètres de l'algorithme XGBoost.
* **Résultat** : Modèle final validé avec une exactitude (Accuracy) globale de 88,40%.

### 03. NLP Alpha Signal — Classification de Filings SEC
* **Problématique** : Analyser la tonalité du discours managérial à partir de 5 000 rapports réglementaires financiers (10-K).
* **Approche** : Fine-tuning du modèle spécialisé FinBERT et extraction sémantique par similarité vectorielle. Pour garantir la validité scientifique, les performances ont été mesurées face à une baseline traditionnelle via le test statistique de Diebold-Mariano avec application de la correction de Benjamini-Hochberg pour le contrôle du taux de fausses découvertes (FDR).
* **Résultat** : Rejet de l'hypothèse nulle (H0) sur l'ensemble des secteurs d'activité (p-values ajustées significatives), confirmant l'apport de l'architecture Transformer.

### 04. Assistant IA Interne — Modèle Hybride RAG + LLM
* **Problématique** : Permettre l'interrogabilité précise et sécurisée d'une base de plus de 1 000 documents techniques complexes.
* **Approche** : Développement d'un pipeline de Retrieval-Augmented Generation (RAG). Indexation des métadonnées et embeddings sémantiques dans une structure FAISS, couplée à un algorithme de clustering thématique. Implémentation d'une couche secondaire de re-ranking (Cross-Encoder) pour optimiser la pertinence du contexte envoyé au LLM.
* **Résultat** : Amélioration de la distribution sémantique atteignant une précision de ciblage final de 92%.

---

## Structure et Exécution

Chaque répertoire racine de projet est configuré de manière autonome et standardisée :

```text
📂 0X_Nom_Du_Projet/
├── 📂 data/          # Fichiers de données ou scripts de simulation
├── 📂 notebooks/     # Analyses exploratoires (EDA) et prototypes
├── 📂 src/           # Code source de production (train, evaluate, etc.)
├── 📂 models/        # Artefacts de modèles sauvegardés et rapports textuels
└── requirements.txt  # Dépendances spécifiques du projet
